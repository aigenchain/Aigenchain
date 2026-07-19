// static/js/imageProviders.js
// AI Defaults → Image Generation card: Image API Providers management.
// Manages image-generation providers (Cloudflare Workers AI, OpenAI-compatible, …)
// through the UI: list, dynamic per-provider form, create/update/delete,
// test connection (real generation), and set active.

function el(id) { return document.getElementById(id); }

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

const IMG_PROVIDER_BADGE = {
  cloudflare: { label: 'Cloudflare', color: '#f6821f' },
  openai_compatible: { label: 'OpenAI-compatible', color: '#10a37f' },
};

function imgProviderBadge(key) {
  const b = IMG_PROVIDER_BADGE[key] || { label: key, color: 'var(--accent, var(--red))' };
  return `<span style="font-size:9px;text-transform:uppercase;letter-spacing:0.5px;padding:1px 5px;border:1px solid color-mix(in srgb, ${b.color} 50%, transparent);border-radius:3px;color:${b.color};background:color-mix(in srgb, ${b.color} 12%, transparent);">${esc(b.label)}</span>`;
}

function imgCard(item) {
  const active = item.is_active;
  const statusDot = active
    ? '<span style="width:8px;height:8px;border-radius:50%;background:var(--color-success,#50fa7b);flex-shrink:0;animation:cookbook-notif-pulse 2s ease-in-out infinite;" title="Active"></span>'
    : '<span style="width:8px;height:8px;border-radius:50%;background:var(--fg);opacity:0.3;flex-shrink:0" title="Inactive"></span>';
  const config = item.config || {};
  const isCf = item.provider === 'cloudflare';
  const _acct = isCf ? config.account_id : config.api_base;
  const _secret = isCf ? config.api_token : config.api_key;
  const _parts = [];
  if (_acct) _parts.push(`<b style="font-weight:600;opacity:0.8;">${isCf ? 'Account' : 'Base'}</b> ${esc(_acct)}`);
  if (_secret) _parts.push(`<b style="font-weight:600;opacity:0.8;">${isCf ? 'API' : 'Key'}</b> ${esc(_secret)}`);
  if (config.model) _parts.push(`<b style="font-weight:600;opacity:0.8;">Model</b> ${esc(config.model)}`);
  const detail = _parts.join(' &middot; ');
  const activeBadge = active
    ? '<span style="font-size:9px;font-weight:700;letter-spacing:0.5px;padding:1px 6px;border-radius:3px;color:#fff;background:var(--color-success,#50fa7b);">ACTIVE</span>'
    : '';
  const setActiveBtn = active ? '' :
    `<button class="admin-btn-sm imgp-activate" data-id="${esc(item.id)}" title="Use this provider for image generation" style="white-space:nowrap;">Set Active</button>`;
  return `<div class="imgp-card" data-id="${esc(item.id)}" data-provider="${esc(item.provider)}" style="display:flex;align-items:center;gap:10px;padding:8px 10px;border:1px solid var(--border);border-radius:8px;background:color-mix(in srgb, var(--fg) 3%, transparent);margin-bottom:6px;cursor:pointer;transition:all 0.15s;" title="Click to edit">
    ${statusDot}
    <div style="flex:1;min-width:0">
      <div style="font-size:12px;font-weight:600;display:flex;align-items:center;gap:6px;flex-wrap:wrap">
        ${esc(item.name || 'Unnamed')} ${imgProviderBadge(item.provider)} ${activeBadge}
      </div>
      <div style="font-size:11px;opacity:0.65;margin-top:2px;line-height:1.4;word-break:break-all">${detail}</div>
    </div>
    <div style="display:flex;gap:5px;align-items:center;flex-shrink:0">
      <button class="admin-btn-sm imgp-test" data-id="${esc(item.id)}" title="Generate a test image to verify the connection" style="white-space:nowrap;">Test</button>
      ${setActiveBtn}
      <button class="admin-btn-sm imgp-edit" data-id="${esc(item.id)}" title="Edit" style="white-space:nowrap;">Edit</button>
      <button class="imgp-del" data-id="${esc(item.id)}" data-name="${esc((item.name || 'this provider').replace(/"/g, '&quot;'))}" title="Remove" style="background:none;border:none;padding:4px;cursor:pointer;color:var(--red);opacity:0.55;display:inline-flex;align-items:center;justify-content:center;">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
      </button>
    </div>
  </div>`;
}

let _imgpDefs = {};
let _imgpListCache = [];

function wireSidebarCloudflare() {
  const cfBtn = el('tool-cloudflare-btn');
  if (!cfBtn) return;
  // The Cloudflare sidebar entry is the gateway to configuring the image
  // provider (account id / api token / model), which is admin-only. Hide it
  // for non-admins; the image model itself is still available to everyone via
  // the model picker once an admin has set up a provider.
  if (!window._isAdmin) { cfBtn.style.display = 'none'; return; }
  if (cfBtn.dataset.wired) return;
  cfBtn.dataset.wired = '1';
  cfBtn.addEventListener('click', () => {
    try {
      if (window.settingsModule && typeof window.settingsModule.open === 'function') {
        window.settingsModule.open('ai');
      }
    } catch (_) {}
  });
}

export function initImageProviders() {
  const listEl = el('image-providers-list');
  const formEl = el('image-providers-form');
  const addBtn = el('image-providers-add-btn');
  if (!listEl || !formEl || !addBtn) return;
  if (addBtn.dataset.wired) return;

  // Sidebar "Cloudflare" entry → jump straight to the Image API Providers
  // settings panel (where the Cloudflare account id / api token / model live).
  wireSidebarCloudflare();

  // Hide "+ Add Provider" while the form is open.
  if (addBtn.parentElement) {
    const wrap = addBtn.parentElement;
    const sync = () => { wrap.style.display = (formEl.style.display && formEl.style.display !== 'none') ? 'none' : ''; };
    new MutationObserver(sync).observe(formEl, { attributes: true, attributeFilter: ['style'] });
    sync();
  }

  addBtn.dataset.wired = '1';
  addBtn.addEventListener('click', () => showForm(null));

  // Refresh list when the AI Defaults tab (where the providers now live) is opened.
  const navBtn = document.querySelector('[data-settings-tab="ai"]');
  if (navBtn && !navBtn.dataset.imgpWired) {
    navBtn.dataset.imgpWired = '1';
    navBtn.addEventListener('click', () => { setTimeout(renderList, 0); });
  }

  renderList();
}

async function loadDefs() {
  if (Object.keys(_imgpDefs).length) return _imgpDefs;
  try {
    const r = await fetch('/api/auth/image-providers/definitions', { credentials: 'same-origin' });
    if (r.ok) { const d = await r.json(); _imgpDefs = d.definitions || {}; }
  } catch (_) {}
  return _imgpDefs;
}

async function renderList() {
  const listEl = el('image-providers-list');
  if (!listEl) return;
  let providers = [];
  try {
    const r = await fetch('/api/auth/image-providers', { credentials: 'same-origin' });
    if (r.ok) { const d = await r.json(); providers = d.providers || []; }
    else if (r.status === 403) {
      listEl.innerHTML = '<div style="padding:12px;opacity:0.6;font-size:12px;text-align:center">Admin access required to manage image providers.</div>';
      return;
    }
  } catch (_) {
    listEl.innerHTML = '<div style="padding:12px;opacity:0.5;font-size:12px;text-align:center">Failed to load providers.</div>';
    return;
  }
  _imgpListCache = providers;
  if (providers.length === 0) {
    listEl.innerHTML = '<div style="padding:14px 12px;opacity:0.85;font-size:12px;text-align:center;line-height:1.5;border:1px dashed var(--border);border-radius:8px;">' +
      '<div style="font-weight:600;margin-bottom:4px;">No image providers configured yet</div>' +
      'Add a provider to generate images from chat and the gallery. ' +
      'Each provider shows only the fields it needs — e.g. Cloudflare asks for <b>Account ID</b>, <b>API Token</b>, and <b>Model</b> ' +
      '(<code>@cf/…/flux-…</code>). Click <b>+ Add Provider</b> above to begin.</div>';
  } else {
    listEl.innerHTML = providers.map(imgCard).join('');
  }
  wireCards();
}

function wireCards() {
  const listEl = el('image-providers-list');
  if (!listEl) return;
  listEl.querySelectorAll('.imgp-card').forEach(card => {
    card.addEventListener('click', (e) => {
      if (e.target.closest('.imgp-test, .imgp-activate, .imgp-edit, .imgp-del')) return;
      showForm(card.dataset.id);
    });
  });
  listEl.querySelectorAll('.imgp-edit').forEach(b =>
    b.addEventListener('click', () => showForm(b.dataset.id)));
  listEl.querySelectorAll('.imgp-test').forEach(b =>
    b.addEventListener('click', () => testProvider(b.dataset.id)));
  listEl.querySelectorAll('.imgp-activate').forEach(b =>
    b.addEventListener('click', () => activateProvider(b.dataset.id)));
  listEl.querySelectorAll('.imgp-del').forEach(b =>
    b.addEventListener('click', () => deleteProvider(b.dataset.id, b.dataset.name)));
}

function fieldRowHtml(field, value) {
  const requiredMark = field.required ? ' <span style="color:var(--red)">*</span>' : '';
  const inputType = field.type === 'password' ? 'password' : 'text';
  const isSecret = field.secret;
  // For secrets that are already saved, surface the masked current value as a
  // placeholder so the admin can confirm it's stored (without revealing it).
  const placeholderVal = (isSecret && value)
    ? `${esc(field.placeholder || '')} (current: ${esc(value)})`
    : (field.placeholder ? esc(field.placeholder) : '');
  const placeholder = placeholderVal ? `placeholder="${placeholderVal}"` : '';
  const shownValue = isSecret ? '' : esc(value || '');
  const help = field.help
    ? `<div style="font-size:10px;opacity:0.5;margin-top:2px;">${esc(field.help)}${isSecret && value ? ' (leave blank to keep current)' : ''}</div>`
    : (isSecret && value ? `<div style="font-size:10px;opacity:0.5;margin-top:2px;">Currently set — leave blank to keep. (current: ${esc(value)})</div>` : '');
  return `<div class="settings-row" style="flex-direction:column;align-items:stretch;gap:3px;">
    <label class="settings-label" style="margin:0;">${esc(field.label)}${requiredMark}</label>
    <input class="settings-select imgp-field" data-key="${esc(field.key)}" data-secret="${isSecret ? '1' : '0'}" data-required="${field.required ? '1' : '0'}" type="${inputType}" ${placeholder} value="${shownValue}" autocomplete="off" style="width:100%;">
    ${help}
  </div>`;
}

async function showForm(editId) {
  const formEl = el('image-providers-form');
  if (!formEl) return;
  await loadDefs();
  const defs = _imgpDefs;
  const defKeys = Object.keys(defs);

  let current = null;
  if (editId && _imgpListCache.length) {
    current = _imgpListCache.find(p => p.id === editId) || null;
  }

  const providerKey = current ? current.provider : (defKeys[0] || '');
  const def = defs[providerKey] || {};

  const providerOpts = defKeys
    .map(k => `<option value="${esc(k)}"${k === providerKey ? ' selected' : ''}>${esc(defs[k].name || k)}</option>`)
    .join('');

  const fieldsHtml = (def.fields || []).map(f => fieldRowHtml(f, current ? (current.config || {})[f.key] : '')).join('');

  const title = current ? 'Edit Provider' : 'Add Provider';
  formEl.innerHTML = `
    <div class="admin-card" style="margin-top:10px;background:color-mix(in srgb, var(--accent, var(--red)) 5%, transparent);">
      <h3 style="margin:0 0 8px;font-size:13px;">${esc(title)}</h3>
      <div class="settings-row" style="flex-direction:column;align-items:stretch;gap:3px;">
        <label class="settings-label" style="margin:0;">Provider</label>
        <select id="imgp-provider-select" class="settings-select" style="width:100%;" ${current ? 'disabled' : ''}>${providerOpts}</select>
      </div>
      <div id="imgp-fields" style="margin-top:8px;">${fieldsHtml}</div>
      <div id="imgp-form-msg" style="font-size:11px;margin-top:6px;min-height:14px;"></div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px;justify-content:flex-end;">
        ${current ? '<button type="button" class="admin-btn-sm" id="imgp-test-btn" style="margin-right:auto;">Test Connection</button>' : ''}
        <button type="button" class="admin-btn-sm" id="imgp-cancel-btn">Cancel</button>
        <button type="button" class="admin-btn-add" id="imgp-save-btn">${current ? 'Save Changes' : 'Add Provider'}</button>
      </div>
    </div>`;
  formEl.style.display = '';

  const sel = el('imgp-provider-select');
  if (sel && !current) {
    sel.addEventListener('change', () => {
      const nd = defs[sel.value] || {};
      el('imgp-fields').innerHTML = (nd.fields || []).map(f => fieldRowHtml(f, '')).join('');
    });
  }
  el('imgp-cancel-btn').addEventListener('click', () => { formEl.style.display = 'none'; formEl.innerHTML = ''; });
  el('imgp-save-btn').addEventListener('click', () => saveForm(editId, providerKey));
  if (current) {
    el('imgp-test-btn').addEventListener('click', async () => {
      const ok = await saveForm(current.id, providerKey);
      if (ok) await testProvider(current.id, false);
    });
  }
}

function collectFields() {
  const config = {};
  document.querySelectorAll('#imgp-fields .imgp-field').forEach(inp => {
    const key = inp.dataset.key;
    const secret = inp.dataset.secret === '1';
    let val = inp.value;
    if (secret && val === '') {
      // sentinel so backend keeps the existing value
      val = '';
    }
    config[key] = val;
  });
  return config;
}

function setMsg(msg, ok) {
  const m = el('imgp-form-msg');
  if (!m) return;
  m.textContent = msg || '';
  m.style.color = ok ? 'var(--color-success,#50fa7b)' : (ok === false ? 'var(--red)' : 'color-mix(in srgb, var(--fg) 45%, transparent)');
}

async function saveForm(editId, providerKey) {
  const config = collectFields();
  const payload = { provider: providerKey, config };
  setMsg('Saving…', null);
  // ── Client-side validation: required fields must be filled. Secrets left
  // blank during an edit are intentionally kept (sentinel) — skip those. ──
  let _firstInvalid = null;
  document.querySelectorAll('#imgp-fields .imgp-field').forEach(inp => {
    const required = inp.dataset.required === '1';
    const secret = inp.dataset.secret === '1';
    inp.style.borderColor = '';
    // When editing an existing provider, a blank secret field means "keep the
    // existing value" — don't treat it as missing. Only block on add (no editId).
    const _keepSecret = secret && editId;
    if (required && !inp.value.trim() && !_keepSecret) {
      inp.style.borderColor = 'var(--red)';
      if (!_firstInvalid) _firstInvalid = inp;
    }
  });
  if (_firstInvalid) {
    _firstInvalid.focus();
    setMsg('Please fill in all required fields (marked with *).', false);
    return false;
  }
  try {
    const url = editId ? `/api/auth/image-providers/${editId}` : '/api/auth/image-providers';
    const method = editId ? 'PUT' : 'POST';
    const r = await fetch(url, {
      method, credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) {
      setMsg(d.detail || d.message || 'Save failed', false);
      return false;
    }
    const formEl = el('image-providers-form');
    formEl.style.display = 'none'; formEl.innerHTML = '';
    await renderList();
    return true;
  } catch (e) {
    setMsg('Save failed: ' + e, false);
    return false;
  }
}

async function activateProvider(id) {
  try {
    const r = await fetch(`/api/auth/image-providers/${id}/activate`, { method: 'POST', credentials: 'same-origin' });
    if (r.ok) await renderList();
  } catch (_) {}
}

async function deleteProvider(id, name) {
  if (!window.styledConfirm) {
    if (!confirm(`Remove "${name}"?`)) return;
  } else if (!await window.styledConfirm(`Remove "${name}"?`, { confirmText: 'Remove', danger: true })) {
    return;
  }
  try {
    const r = await fetch(`/api/auth/image-providers/${id}`, { method: 'DELETE', credentials: 'same-origin' });
    if (r.ok) await renderList();
  } catch (_) {}
}

async function testProvider(id, fromForm) {
  const card = document.querySelector(`.imgp-test[data-id="${CSS.escape(id)}"]`);
  const origLabel = card ? card.textContent : 'Test';
  if (card) { card.disabled = true; card.textContent = 'Testing…'; }
  // If testing from the (now hidden) edit form, surfaces the result via the
  // list card; otherwise straight to the list card.
  try {
    const r = await fetch(`/api/auth/image-providers/${id}/test`, { method: 'POST', credentials: 'same-origin' });
    const d = await r.json().catch(() => ({}));
    const ok = !!d.ok;
    flashTestResult(id, ok, d.message || (ok ? 'OK' : 'Failed'));
    if (fromForm) setMsg(d.message || (ok ? 'Connection OK' : 'Test failed'), ok);
  } catch (e) {
    flashTestResult(id, false, 'Test failed: ' + e);
    if (fromForm) setMsg('Test failed: ' + e, false);
  } finally {
    if (card) { card.disabled = false; card.textContent = origLabel; }
  }
}

function flashTestResult(id, ok, message) {
  const card = document.querySelector(`.imgp-card[data-id="${CSS.escape(id)}"]`);
  if (!card) return;
  let note = card.querySelector('.imgp-test-note');
  if (!note) {
    note = document.createElement('div');
    note.className = 'imgp-test-note';
    note.style.cssText = 'font-size:10px;margin-top:4px;';
    card.appendChild(note);
  }
  note.textContent = (ok ? '✓ ' : '✗ ') + message;
  note.style.color = ok ? 'var(--color-success,#50fa7b)' : 'var(--red)';
  setTimeout(() => { note.textContent = ''; }, 8000);
}
