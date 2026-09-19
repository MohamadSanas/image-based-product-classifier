// ============================================================
// SmartMart AI Checkout — Frontend Logic
// EC9570 Digital Image Processing
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const previewThumbnail = document.getElementById('preview-thumbnail');
  const thumbImg = document.getElementById('thumb-img');
  const clearFileBtn = document.getElementById('clear-file-btn');
  const sampleCarousel = document.getElementById('sample-carousel');
  const sampleCount = document.getElementById('sample-count');
  
  const confSlider = document.getElementById('conf-slider');
  const confVal = document.getElementById('conf-val');
  const denoiseToggle = document.getElementById('denoise-toggle');
  const claheToggle = document.getElementById('clahe-toggle');
  const scanBtn = document.getElementById('scan-btn');
  const scanSpinner = document.getElementById('scan-spinner');

  const viewTabs = document.getElementById('view-tabs');
  const totalItemsBadge = document.getElementById('total-items-badge');
  const meanConfBadge = document.getElementById('mean-conf-badge');
  const isolatedBadgeCount = document.getElementById('isolated-badge-count');

  const emptyState = document.getElementById('empty-state');
  const resultAnnotatedImage = document.getElementById('result-annotated-image');
  const isolatedGrid = document.getElementById('isolated-grid');
  
  const chartBar = document.getElementById('chart-bar');
  const chartPie = document.getElementById('chart-pie');
  const chartConf = document.getElementById('chart-conf');

  const receiptItems = document.getElementById('receipt-items');
  const receiptTotalUnits = document.getElementById('receipt-total-units');
  const receiptTotalCats = document.getElementById('receipt-total-cats');
  const receiptAccuracy = document.getElementById('receipt-accuracy');
  const receiptDate = document.getElementById('receipt-date');
  const receiptTime = document.getElementById('receipt-time');
  const downloadReportBtn = document.getElementById('download-report-btn');
  const printReceiptBtn = document.getElementById('print-receipt-btn');
  const toast = document.getElementById('toast');

  // State
  let currentFile = null;
  let selectedSampleName = null;
  let latestReportText = "";

  // Set real date/time on receipt
  const now = new Date();
  receiptDate.textContent = now.toLocaleDateString();
  receiptTime.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  // Slider change
  confSlider.addEventListener('input', (e) => {
    confVal.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // ──────────────────────────────────────────────────────────
  // 1. Fetch Sample Images
  // ──────────────────────────────────────────────────────────
  async function loadSamples() {
    try {
      const res = await fetch('/api/samples');
      const data = await res.json();
      if (!data.samples || data.samples.length === 0) {
        sampleCarousel.innerHTML = `<div class="loading-chip">No samples found</div>`;
        return;
      }

      sampleCount.textContent = `${data.samples.length} Available`;
      sampleCarousel.innerHTML = '';

      data.samples.forEach((name, idx) => {
        const chip = document.createElement('div');
        chip.className = 'sample-chip';
        chip.title = name;
        chip.innerHTML = `
          <span style="font-size: 14px;">🛍️</span>
          <span>Sample ${idx + 1}</span>
        `;

        chip.addEventListener('click', () => {
          document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('active'));
          chip.classList.add('active');
          selectSample(name);
        });

        sampleCarousel.appendChild(chip);
      });

      // Auto-select first sample for instant delight
      if (data.samples.length > 0) {
        const firstChip = sampleCarousel.children[0];
        firstChip.classList.add('active');
        selectSample(data.samples[0]);
      }
    } catch (err) {
      console.error("Failed to load samples:", err);
      sampleCarousel.innerHTML = `<div class="loading-chip">Failed to load samples</div>`;
    }
  }

  function selectSample(sampleName) {
    selectedSampleName = sampleName;
    currentFile = null;
    thumbImg.src = `/api/sample-image/${encodeURIComponent(sampleName)}`;
    previewThumbnail.classList.remove('hidden');
    showToast(`Selected sample basket: ${sampleName}`);
  }

  // ──────────────────────────────────────────────────────────
  // 2. File Upload & Drag-and-Drop
  // ──────────────────────────────────────────────────────────
  dropzone.addEventListener('click', (e) => {
    if (e.target !== clearFileBtn) {
      fileInput.click();
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleUserFile(e.target.files[0]);
    }
  });

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleUserFile(e.dataTransfer.files[0]);
    }
  });

  clearFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    currentFile = null;
    selectedSampleName = null;
    fileInput.value = '';
    previewThumbnail.classList.add('hidden');
    document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('active'));
  });

  function handleUserFile(file) {
    currentFile = file;
    selectedSampleName = null;
    document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('active'));

    const reader = new FileReader();
    reader.onload = (evt) => {
      thumbImg.src = evt.target.result;
      previewThumbnail.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
    showToast(`Loaded: ${file.name}`);
  }

  // ──────────────────────────────────────────────────────────
  // 3. Tab Switching
  // ──────────────────────────────────────────────────────────
  viewTabs.addEventListener('click', (e) => {
    if (!e.target.classList.contains('tab-btn')) return;
    const viewName = e.target.dataset.view;

    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    e.target.classList.add('active');

    document.querySelectorAll('.view-content').forEach(view => {
      view.classList.remove('active');
    });
    document.getElementById(`view-${viewName}`).classList.add('active');
  });

  // ──────────────────────────────────────────────────────────
  // 4. Run Detection Pipeline
  // ──────────────────────────────────────────────────────────
  scanBtn.addEventListener('click', async () => {
    if (!currentFile && !selectedSampleName) {
      showToast("Please choose a sample or upload an image first!", true);
      return;
    }

    setLoading(true);
    showToast("Processing basket through YOLOv8 pipeline...");

    const formData = new FormData();
    formData.append('conf', confSlider.value);
    formData.append('denoise', denoiseToggle.checked);
    formData.append('clahe', claheToggle.checked);

    if (currentFile) {
      formData.append('file', currentFile);
    } else {
      formData.append('sample_name', selectedSampleName);
    }

    try {
      const res = await fetch('/api/detect', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Scanning failed.");
      }

      const result = await res.json();
      renderResults(result);
      showToast(`Scan complete! ${result.total_detections} items detected.`);
    } catch (err) {
      console.error(err);
      showToast(err.message, true);
    } finally {
      setLoading(false);
    }
  });

  function setLoading(isLoading) {
    scanBtn.disabled = isLoading;
    if (isLoading) {
      scanSpinner.classList.remove('hidden');
      scanBtn.querySelector('.btn-label').textContent = 'Scanning Basket...';
    } else {
      scanSpinner.classList.add('hidden');
      scanBtn.querySelector('.btn-label').textContent = 'Run Smart Checkout Scan';
    }
  }

  // ──────────────────────────────────────────────────────────
  // 5. Render Scan Results
  // ──────────────────────────────────────────────────────────
  function renderResults(data) {
    // 1. Header Badges
    totalItemsBadge.textContent = data.total_detections;
    meanConfBadge.textContent = `${(data.mean_confidence * 100).toFixed(1)}%`;
    isolatedBadgeCount.textContent = data.crops ? data.crops.length : 0;

    // 2. Annotated Viewport
    emptyState.classList.add('hidden');
    resultAnnotatedImage.classList.remove('hidden');
    resultAnnotatedImage.src = data.annotated_image;

    // 3. Isolated Products (Segmentation ROIs)
    isolatedGrid.innerHTML = '';
    if (data.crops && data.crops.length > 0) {
      data.crops.forEach((crop, idx) => {
        const card = document.createElement('div');
        card.className = 'product-crop-card';
        card.innerHTML = `
          <div class="crop-img-wrap">
            <img src="${crop.image_data}" alt="${crop.product_name}">
          </div>
          <div class="crop-meta">
            <h5 title="${crop.product_name}">${idx + 1}. ${crop.product_name}</h5>
            <span class="crop-category-badge">${crop.category}</span>
            <div class="crop-conf-bar">
              <span>Confidence</span>
              <strong>${(crop.confidence * 100).toFixed(1)}%</strong>
            </div>
          </div>
        `;
        isolatedGrid.appendChild(card);
      });
    } else {
      isolatedGrid.innerHTML = `<div class="empty-state-small">No individual products were isolated.</div>`;
    }

    // 4. Charts
    if (data.charts) {
      chartBar.src = data.charts.bar || '';
      chartPie.src = data.charts.pie || '';
      chartConf.src = data.charts.conf || '';
    }

    // 5. POS Digital Receipt
    receiptTotalUnits.textContent = data.total_detections;
    receiptTotalCats.textContent = data.num_categories;
    receiptAccuracy.textContent = `${(data.mean_confidence * 100).toFixed(1)}%`;

    receiptItems.innerHTML = '';
    if (data.category_summary && data.category_summary.length > 0) {
      data.category_summary.forEach(cat => {
        const row = document.createElement('div');
        row.className = 'receipt-item-row';
        row.innerHTML = `
          <div class="item-left">
            <span class="item-name">${cat.category}</span>
            <span class="item-category">${cat.products ? cat.products.slice(0, 2).join(', ') : ''}</span>
          </div>
          <div class="item-qty-share">
            <span>x${cat.count}</span>
            <span class="item-share">${cat.share}%</span>
          </div>
        `;
        receiptItems.appendChild(row);
      });
    } else {
      receiptItems.innerHTML = `<div class="receipt-empty">No products detected.</div>`;
    }

    latestReportText = data.report_text || "";
    downloadReportBtn.disabled = !latestReportText;
    printReceiptBtn.disabled = false;
  }

  // ──────────────────────────────────────────────────────────
  // 6. Action Handlers (Download Report & Print Receipt)
  // ──────────────────────────────────────────────────────────
  downloadReportBtn.addEventListener('click', () => {
    if (!latestReportText) return;
    const blob = new Blob([latestReportText], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `SmartMart_Checkout_Report_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    showToast("Downloaded checkout report!");
  });

  printReceiptBtn.addEventListener('click', () => {
    window.print();
  });

  function showToast(msg, isError = false) {
    toast.textContent = msg;
    toast.style.borderColor = isError ? 'var(--accent-red)' : 'var(--border-glow)';
    toast.classList.remove('hidden');
    setTimeout(() => {
      toast.classList.add('hidden');
    }, 3500);
  }

  // Initialize
  loadSamples();
});
