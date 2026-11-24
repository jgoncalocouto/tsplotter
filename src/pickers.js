/** Searchable variable pickers backed by the original, portable native selects. */
const controllers = new Set();
const bySelect = new WeakMap();
const attachedDocuments = new WeakSet();
let nextId = 0;

/** Search labels without changing original values, order or diacritics in the dataset. */
export function filterPickerOptions(options, query = '') {
  const normalize = value => String(value ?? '').normalize('NFKD').replace(/[\u0300-\u036f]/g, '').toLocaleLowerCase();
  const terms = normalize(query).trim().split(/\s+/).filter(Boolean);
  return options.filter(option => !option.hidden && terms.every(term => normalize(option.label).includes(term)));
}
export function pickerSummary(options, limit = 3) {
  const selected = options.filter(option => option.selected);
  return {selected, chips: selected.slice(0, Math.max(0, limit)), remaining: Math.max(0, selected.length - limit), count: selected.length};
}
/** Return the next source option index, wrapping only among enabled search matches. */
export function nextPickerOption(options, activeIndex, direction = 1) {
  const enabled = options.filter(option => !option.disabled && !option.hidden);
  if (!enabled.length) return undefined;
  const position = enabled.findIndex(option => option.index === activeIndex);
  if (position < 0) return enabled[direction < 0 ? enabled.length - 1 : 0].index;
  return enabled[(position + (direction < 0 ? -1 : 1) + enabled.length) % enabled.length].index;
}
export function readPickerOptions(select) {
  return Array.from(select.options, (option, index) => ({
    index, value: option.value, label: option.label || option.textContent || option.value,
    selected: Boolean(option.selected), hidden: Boolean(option.hidden),
    disabled: Boolean(option.disabled || option.parentElement?.tagName === 'OPTGROUP' && option.parentElement.disabled),
  }));
}
function dispatchChange(select) {
  const EventClass = select.ownerDocument?.defaultView?.Event || globalThis.Event;
  select.dispatchEvent(new EventClass('change', {bubbles: true}));
}
/** Native change events remain the single integration point with configuration/state capture. */
export function updatePickerSelection(select, index, selected, {notify = true} = {}) {
  if (select.disabled) return false;
  const option = select.options[index];
  if (!option || option.disabled || option.parentElement?.tagName === 'OPTGROUP' && option.parentElement.disabled || Boolean(option.selected) === Boolean(selected)) return false;
  option.selected = Boolean(selected);
  if (notify) dispatchChange(select);
  return true;
}
export function clearPickerSelection(select, {notify = true} = {}) {
  if (select.disabled) return false;
  let changed = false;
  for (let index = 0; index < select.options.length; index++) changed = updatePickerSelection(select, index, false, {notify: false}) || changed;
  if (changed && notify) dispatchChange(select);
  return changed;
}
function element(document, tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function selectLabel(select) {
  const label = select.labels?.[0] || select.closest?.('label');
  const text = label && Array.from(label.childNodes).filter(node => node.nodeType === 3).map(node => node.textContent).join(' ').trim();
  return select.getAttribute('aria-label') || text || 'Signals';
}
function cleanupDetached() {
  for (const controller of controllers) {
    if (!controller.select.isConnected || !controller.wrapper.isConnected) {
      controller.observer?.disconnect();
      controller.select.removeEventListener('change', controller.nativeChange);
      controllers.delete(controller);
      bySelect.delete(controller.select);
    }
  }
}
function attachDocument(document) {
  if (attachedDocuments.has(document)) return;
  attachedDocuments.add(document);
  for (const type of ['pointerdown', 'click', 'focusin']) document.addEventListener(type, event => {
    cleanupDetached();
    for (const controller of controllers) {
      if (controller.select.ownerDocument === document && controller.open && !controller.wrapper.contains(event.target)) controller.close(false);
    }
  }, true);
}
/** Close every open popup, for example before a large calculation disables the form. */
export function closePickers({restoreFocus = false} = {}) {
  cleanupDetached();
  for (const controller of controllers) controller.close(restoreFocus);
}
function createPicker(select) {
  const document = select.ownerDocument;
  attachDocument(document);
  const label = selectLabel(select), multiple = select.multiple;
  const token = `${select.id || 'signal'}-picker-${++nextId}`;
  const wrapper = element(document, 'div', 'signal-picker');
  wrapper.dataset.pickerMode = multiple ? 'multiple' : 'single';
  const trigger = element(document, 'button', 'picker-trigger');
  trigger.type = 'button'; trigger.setAttribute('aria-haspopup', 'listbox'); trigger.setAttribute('aria-expanded', 'false'); trigger.setAttribute('aria-controls', `${token}-list`);
  const popover = element(document, 'div', 'picker-popover'); popover.hidden = true;
  const search = element(document, 'input', 'picker-search');
  search.type = 'search'; search.autocomplete = 'off'; search.placeholder = 'Search variables…'; search.setAttribute('aria-label', `Search ${label.toLocaleLowerCase()}`); search.setAttribute('aria-controls', `${token}-list`);
  const list = element(document, 'div', 'picker-options'); list.id = `${token}-list`; list.setAttribute('role', 'listbox'); list.setAttribute('aria-label', label); list.setAttribute('aria-multiselectable', String(multiple));
  const empty = element(document, 'div', 'picker-empty', 'No matching signals'); empty.setAttribute('role', 'status'); empty.hidden = true;
  const footer = element(document, 'div', 'picker-footer');
  footer.hidden = !multiple;
  const count = element(document, 'span', 'picker-count'); count.setAttribute('aria-live', 'polite');
  const clear = element(document, 'button', 'picker-clear', 'Clear selection'); clear.type = 'button';
  footer.append(count, clear); popover.append(search, list, empty, footer); wrapper.append(trigger, popover);
  select.classList.add('native-control'); select.tabIndex = -1; select.setAttribute('aria-hidden', 'true'); select.after(wrapper);
  const controller = {select, wrapper, trigger, popover, search, open: false, options: [], activeIndex: undefined, observer: undefined, nativeChange: undefined, close: undefined, sync: undefined};
  const buttons = new Map();
  const visible = () => filterPickerOptions(controller.options, search.value);
  function close(restore = false) {
    if (!controller.open) return;
    controller.open = false; popover.hidden = true; trigger.setAttribute('aria-expanded', 'false');
    if (restore && !select.disabled && trigger.isConnected) trigger.focus();
  }
  controller.close = close;
  function focusOption(index) {
    if (index === undefined || !buttons.has(index)) return;
    controller.activeIndex = index;
    for (const [key, button] of buttons) button.tabIndex = key === index ? 0 : -1;
    const button = buttons.get(index); button.focus(); button.scrollIntoView?.({block: 'nearest'});
  }
  function renderMatches() {
    const matches = visible();
    const matching = new Set(matches.map(option => option.index));
    const next = matches.some(option => option.index === controller.activeIndex && !option.disabled) ? controller.activeIndex : nextPickerOption(matches, undefined, 1);
    controller.activeIndex = next;
    for (const [index, button] of buttons) {button.hidden = !matching.has(index); button.tabIndex = index === next ? 0 : -1;}
    empty.hidden = matches.length > 0;
  }
  function renderTrigger() {
    trigger.replaceChildren();
    const summary = pickerSummary(controller.options);
    const chips = element(document, 'span', 'picker-values');
    if (!summary.count) chips.append(element(document, 'span', 'picker-placeholder', 'Select variables…'));
    if (!multiple && summary.count) {
      const value = element(document, 'span', 'picker-single-value', summary.selected[0].label);
      value.title = summary.selected[0].label; chips.append(value);
    }
    for (const option of multiple ? summary.chips : []) {
      const chip = element(document, 'span', 'picker-chip'); chip.title = option.label;
      chip.append(element(document, 'span', 'picker-chip-label', option.label));
      if (!option.disabled) {
        const remove = element(document, 'span', 'picker-chip-remove', '×'); remove.dataset.pickerRemove = String(option.index); remove.setAttribute('aria-hidden', 'true'); chip.append(remove);
      }
      chips.append(chip);
    }
    if (multiple && summary.remaining) chips.append(element(document, 'span', 'picker-overflow', `+${summary.remaining}`));
    const caret = element(document, 'span', 'picker-caret', '⌄'); caret.setAttribute('aria-hidden', 'true');
    trigger.append(chips, caret);
    trigger.setAttribute('aria-label', `${label}: ${summary.count ? `${multiple ? `${summary.count} selected, ` : ''}${summary.selected.map(option => option.label).join(', ')}` : 'no selection'}`);
    count.textContent = `${summary.count} selected`;
    clear.disabled = select.disabled || !summary.selected.some(option => !option.disabled);
  }
  function setSelection(index, value) {
    if (!multiple) {value = true; close(true);}
    if (!updatePickerSelection(select, index, value, {notify: false})) return;
    sync(); // Update before the app's bubbling handler can replace this picker.
    dispatchChange(select);
  }
  function sync() {
    const active = document.activeElement;
    const focusedOption = active?.closest?.('.picker-option');
    controller.options = readPickerOptions(select);
    const signature = controller.options.map(option => `${option.index}\u0000${option.label}\u0000${option.value}\u0000${option.disabled}\u0000${option.hidden}`).join('\u0001');
    if (signature !== controller.signature) {
      controller.signature = signature; buttons.clear(); list.replaceChildren();
      for (const option of controller.options) {
        const button = element(document, 'button', 'picker-option'); button.type = 'button'; button.id = `${token}-option-${option.index}`; button.dataset.pickerIndex = String(option.index); button.setAttribute('role', 'option'); button.tabIndex = -1;
        const tick = element(document, 'span', 'picker-tick', '✓'); tick.setAttribute('aria-hidden', 'true');
        button.append(tick, element(document, 'span', 'picker-option-label', option.label));
        button.addEventListener('click', () => setSelection(option.index, !select.options[option.index]?.selected));
        button.addEventListener('focus', () => {controller.activeIndex = option.index;});
        button.addEventListener('keydown', event => {
          if (event.key === 'Escape') {event.preventDefault(); event.stopPropagation(); close(true);}
          else if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
            event.preventDefault();
            const matches = visible();
            focusOption(event.key === 'Home' ? nextPickerOption(matches, undefined, 1) : event.key === 'End' ? nextPickerOption(matches, undefined, -1) : nextPickerOption(matches, option.index, event.key === 'ArrowUp' ? -1 : 1));
          } else if (event.key === 'Enter' || event.key === ' ') {event.preventDefault(); setSelection(option.index, !select.options[option.index]?.selected);}
        });
        buttons.set(option.index, button); list.append(button);
      }
    }
    trigger.disabled = select.disabled; search.disabled = select.disabled;
    for (const option of controller.options) {
      const button = buttons.get(option.index); button.disabled = select.disabled || option.disabled; button.setAttribute('aria-selected', String(option.selected)); button.classList.toggle('selected', option.selected);
    }
    renderTrigger(); renderMatches();
    if (select.disabled) close(false);
    // Rebuilding a list after native options change must not strand keyboard focus.
    else if (controller.open && focusedOption && wrapper.contains(focusedOption)) focusOption(controller.activeIndex);
  }
  controller.sync = sync;
  function open(focus = 'search') {
    if (select.disabled) return;
    for (const other of controllers) if (other !== controller) other.close(false);
    controller.open = true; popover.hidden = false; trigger.setAttribute('aria-expanded', 'true');
    search.value = ''; renderMatches();
    if (focus === 'search') search.focus();
    else focusOption((!multiple ? visible().find(option => option.selected && !option.disabled)?.index : undefined) ?? nextPickerOption(visible(), undefined, focus === 'last' ? -1 : 1));
  }
  trigger.addEventListener('click', event => {
    const removal = event.target.closest?.('[data-picker-remove]');
    if (removal) {event.preventDefault(); event.stopPropagation(); setSelection(Number(removal.dataset.pickerRemove), false); return;}
    controller.open ? close(false) : open();
  });
  trigger.addEventListener('keydown', event => {
    if (event.key === 'Escape' && controller.open) {event.preventDefault(); event.stopPropagation(); close(true);}
    else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {event.preventDefault(); open(event.key === 'ArrowUp' ? 'last' : 'first');}
    else if (event.key === 'Enter' || event.key === ' ') {event.preventDefault(); controller.open ? close(false) : open();}
    else if (multiple && (event.key === 'Backspace' || event.key === 'Delete') && !controller.open) {
      event.preventDefault(); const last = controller.options.filter(option => option.selected && !option.disabled).at(-1); if (last) setSelection(last.index, false);
    } else if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {
      event.preventDefault(); open(); search.value = event.key; renderMatches();
    }
  });
  search.addEventListener('input', renderMatches);
  search.addEventListener('keydown', event => {
    if (event.key === 'Escape') {event.preventDefault(); event.stopPropagation(); close(true);}
    else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {event.preventDefault(); focusOption(nextPickerOption(visible(), undefined, event.key === 'ArrowUp' ? -1 : 1));}
    else if (event.key === 'Enter') {event.preventDefault(); const index = nextPickerOption(visible(), undefined, 1); if (index !== undefined) setSelection(index, !select.options[index]?.selected);}
  });
  clear.addEventListener('click', () => {if (clearPickerSelection(select, {notify: false})) {sync(); dispatchChange(select);} search.focus();});
  controller.nativeChange = sync; select.addEventListener('change', sync);
  const MutationObserverClass = document.defaultView?.MutationObserver || globalThis.MutationObserver;
  if (MutationObserverClass) {
    controller.observer = new MutationObserverClass(sync);
    controller.observer.observe(select, {childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ['disabled', 'selected', 'label', 'value', 'hidden']});
  }
  bySelect.set(select, controller); controllers.add(controller); sync();
  return controller;
}
/** Safe to call after every view/configuration/dataset render; existing controls are synchronized once. */
export function enhancePickers(root = document) {
  cleanupDetached();
  const selector = 'select[data-variable],select[data-key][multiple],select[data-panel-key][multiple],select[data-trend-subplot][multiple]';
  const selects = [...root.querySelectorAll(selector)];
  if (root.matches?.(selector)) selects.unshift(root);
  return selects.map(select => {
    const existing = bySelect.get(select);
    if (existing) {existing.sync(); return existing;}
    return createPicker(select);
  });
}
