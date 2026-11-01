// The Chat's transcript: the open conversation as the app sends it (Transcript.swift). The app
// calls setup once the page has loaded, show with the whole conversation when it changes, and
// update with the reply that streams; the page sends back what to copy, and Ask Again.
'use strict';

const transcript = (() => {
  const messages = document.getElementById('messages');
  const latest = document.getElementById('latest');
  const scroller = document.scrollingElement;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const post = message => window.webkit?.messageHandlers.transcript.postMessage(message);

  // MARK: Markdown

  const escape = text => text.replace(/[&<>"']/g, character => `&#${character.charCodeAt(0)};`);
  const web = url => /^(https?|mailto):/i.test(url ?? '');
  const copyGlyphs = '<span class="glyph copy"></span><span class="glyph check"></span>';
  const bullets = ['•', '◦', '▪'];

  // A reply's Markdown: no raw HTML, its line breaks kept, only web and mail links, and an image as
  // a link to it, so that a reply never loads anything by itself.
  const markdown = window.markdownit ? configured(window.markdownit({html: false, breaks: true})) : null;

  function configured(markdown) {
    // Every link reaches the rule below, which keeps web and mail links and the text of others.
    markdown.validateLink = () => true;
    markdown.core.ruler.push('splash', state => {
      let cell = false;
      for (const token of state.tokens) {
        // A list item's text is a paragraph, in a tight list or a loose one.
        if (token.type === 'paragraph_open' || token.type === 'paragraph_close') token.hidden = false;
        if (token.type === 'th_open' || token.type === 'td_open') cell = true;
        if (token.type === 'th_close' || token.type === 'td_close') cell = false;
        if (token.type !== 'inline') continue;
        token.children = links(token.children, state);
        if (cell) token.children = lineBreaks(token.children, state);
      }
    });
    const rules = markdown.renderer.rules;
    rules.fence = (tokens, index) =>
      code(tokens[index].content, markdown.utils.unescapeAll(tokens[index].info).trim().split(/\s+/)[0]);
    rules.code_block = (tokens, index) => code(tokens[index].content, '');
    // Markers in a column of their own, as wide as the widest, so that the items' text lines up;
    // numbered from the list's start, bullets by how deep the list is.
    rules.bullet_list_open = (tokens, index, options, env) => {
      env.lists.push({ordered: false});
      return '<ul>';
    };
    rules.ordered_list_open = (tokens, index, options, env) => {
      env.lists.push({ordered: true, next: Number(tokens[index].attrGet('start') ?? 1)});
      return '<ol>';
    };
    rules.bullet_list_close = rules.ordered_list_close = (tokens, index, options, env) =>
      env.lists.pop().ordered ? '</ol>' : '</ul>';
    rules.list_item_open = (tokens, index, options, env) => {
      const list = env.lists.at(-1);
      const marker = list.ordered ? `${list.next++}.` : bullets[(env.lists.length - 1) % bullets.length];
      return `<li><span class="marker">${marker}</span><div class="item">`;
    };
    rules.list_item_close = () => '</div></li>';
    // A table scrolls sideways when it is wider than the reply; its cells wrap at a width of their
    // own and line up at their start.
    rules.table_open = (tokens, index, options, env, self) => '<div class="table">' + self.renderToken(tokens, index, options);
    rules.table_close = (tokens, index, options, env, self) => self.renderToken(tokens, index, options) + '</div>';
    for (const cell of ['th', 'td']) {
      rules[`${cell}_open`] = (tokens, index, options, env, self) => {
        tokens[index].attrs = null;
        return self.renderToken(tokens, index, options) + '<div>';
      };
      rules[`${cell}_close`] = (tokens, index, options, env, self) => '</div>' + self.renderToken(tokens, index, options);
    }
    return markdown;
  }

  // The inline tokens with only web and mail links left links, and images as links to them.
  function links(children, state) {
    const kept = [];
    const dropped = [];
    for (const token of children) {
      if (token.type === 'link_open') {
        dropped.push(!web(token.attrGet('href')));
        if (!dropped.at(-1)) kept.push(token);
      } else if (token.type === 'link_close') {
        if (!dropped.pop()) kept.push(token);
      } else if (token.type === 'image') {
        const text = Object.assign(new state.Token('text', '', 0), {content: state.md.renderer.renderInlineAsText(token.children, {}, {})});
        // An image in a link is that link's text.
        if (web(token.attrGet('src')) && !dropped.includes(false)) {
          const open = new state.Token('link_open', 'a', 1);
          open.attrs = [['href', token.attrGet('src')]];
          kept.push(open, text, new state.Token('link_close', 'a', -1));
        } else {
          kept.push(text);
        }
      } else {
        kept.push(token);
      }
    }
    return kept;
  }

  // A table cell's <br>s as line breaks, as GitHub shows them.
  function lineBreaks(children, state) {
    return children.flatMap(token => token.type !== 'text' ? [token] : token.content.split(/<br\s*\/?>/i).flatMap((part, index) => [
      ...(index > 0 ? [new state.Token('hardbreak', 'br', 0)] : []),
      Object.assign(new state.Token('text', '', 0), {content: part}),
    ]));
  }

  // Code, outlined, under its language and Copy; highlighted when highlight.js knows the language.
  function code(source, language) {
    const text = source.replace(/\n$/, '');
    const highlighted = language && window.hljs?.getLanguage(language)
      ? window.hljs.highlight(text, {language, ignoreIllegals: true}).value
      : escape(text);
    return `<figure class="code"><figcaption><span>${escape(language)}</span>` +
      `<button type="button" class="hover copy-code" title="Copy" aria-label="Copy Code">${copyGlyphs}</button></figcaption>` +
      `<pre><code>${highlighted}</code></pre></figure>`;
  }

  function render(text) {
    try {
      return markdown.render(text, {lists: []});
    } catch {
      // Never nothing: the text as it came, should Markdown be missing or fail.
      return `<p>${escape(text)}</p>`;
    }
  }

  // Draws html into container block by block, replacing only the blocks that changed: a streaming
  // reply redraws its last blocks, and the others keep their selection and their place.
  function patch(container, html) {
    const fresh = document.createElement('div');
    fresh.innerHTML = html;
    const blocks = [...fresh.children];
    const old = [...container.children];
    blocks.forEach((block, index) => {
      if (index >= old.length) container.append(block);
      else if (!old[index].isEqualNode(block)) old[index].replaceWith(block);
    });
    for (const extra of old.slice(blocks.length)) extra.remove();
  }

  // MARK: Messages

  // The messages drawn, by id: their element and the message as last drawn.
  const views = new Map();
  // The reader's choice to open or close a reply's thinking, by id.
  const expanded = new Map();
  let conversation = null;
  let askAgain = {enabled: false, help: ''};

  const replyMarkup =
    '<div class="line"><button type="button" class="hover toggle"><span class="glyph chevron"></span>' +
    '<span class="verb"></span><span class="facts"></span></button>' +
    '<div class="thought"><div class="fold"><div class="reasoning"><div class="text"><div></div></div></div></div></div></div>' +
    '<div class="markdown"></div>' +
    '<p class="error"><span class="glyph warning"></span><span class="message"></span></p>' +
    '<div class="actions"><button type="button" class="hover copy-reply" title="Copy" aria-label="Copy Reply">' + copyGlyphs +
    '</button><button type="button" class="hover again" aria-label="Ask Again"><span class="glyph again"></span></button>' +
    '<span class="details"></span></div>';

  function draw(message) {
    let view = views.get(message.id);
    if (!view) {
      const element = document.createElement('article');
      element.className = message.role === 'user' ? 'user' : 'reply';
      element.dataset.id = message.id;
      element.innerHTML = message.role === 'user' ? '<div class="images"></div><p class="bubble"></p>' : replyMarkup;
      view = {element, message: null};
      views.set(message.id, view);
    }
    if (!same(view.message, message)) {
      (message.role === 'user' ? drawUser : drawReply)(view.element, message, view.message);
      view.message = message;
    }
    return view.element;
  }

  function same(a, b) {
    return a !== null && a.text === b.text && a.reasoning === b.reasoning && a.error === b.error && a.live === b.live &&
      a.details === b.details && a.line?.verb === b.line?.verb && a.line?.facts === b.line?.facts &&
      a.line?.since === b.line?.since && sameImages(a, b);
  }

  const sameImages = (a, b) => a.images.length === b.images.length && a.images.every((image, index) => image === b.images[index]);

  // The user's images over their text, at the column's end.
  function drawUser(element, message, previous) {
    const images = element.querySelector('.images');
    if (!previous || !sameImages(previous, message)) {
      images.replaceChildren(...message.images.map(url => Object.assign(document.createElement('img'), {src: url, alt: ''})));
    }
    images.hidden = message.images.length === 0;
    const bubble = element.querySelector('.bubble');
    bubble.textContent = message.text;
    bubble.hidden = message.text === '';
  }

  // A reply: its line, its Markdown, why it stopped short, and its actions, which keep their place
  // and show under the pointer and on the last reply, but never while it streams.
  function drawReply(element, message, previous) {
    drawLine(element, message);
    const body = element.querySelector('.markdown');
    if (message.text !== previous?.text) patch(body, render(message.text));
    body.hidden = message.text === '';
    const error = element.querySelector('.error');
    error.querySelector('.message').textContent = message.error ?? '';
    error.hidden = message.error === null;
    element.classList.toggle('live', message.live);
    element.querySelector('.details').textContent = message.details;
    element.querySelector('.actions').setAttribute('aria-hidden', message.live);
  }

  // The line over a reply: its verb while it waits, which shows half a second into the wait, so
  // that a quick answer doesn't flash it; what the thinking took after. It opens to the thinking,
  // of which it shows the latest lines while the model thinks.
  function drawLine(element, message) {
    const line = element.querySelector('.line');
    line.hidden = message.line === null;
    if (message.line === null) return count(line, null);
    const reasoned = message.reasoning !== '';
    const thinking = message.live && message.text === '' && reasoned;
    const open = reasoned && (expanded.get(message.id) ?? thinking);
    line.classList.toggle('waiting', message.live && !reasoned);
    line.classList.toggle('reasoned', reasoned);
    line.classList.toggle('thinking', thinking);
    line.classList.toggle('open', open);
    const toggle = line.querySelector('.toggle');
    if (reasoned) toggle.setAttribute('aria-expanded', open);
    else toggle.removeAttribute('aria-expanded');
    line.querySelector('.verb').textContent = message.line.verb;
    if (message.line.since === null) facts(line, message.line.facts);
    count(line, message.line.since);
    const text = line.querySelector('.text > div');
    if (text.textContent !== message.reasoning) text.textContent = message.reasoning;
    // Measured once the reply is in the page, as a reply shown with its conversation isn't yet.
    queueMicrotask(() => line.classList.toggle('tall', thinking && text.offsetHeight > 96));
  }

  function facts(line, words) {
    const span = line.querySelector('.facts');
    span.textContent = words ? `· ${words}` : '';
    span.hidden = !words;
  }

  // Whole seconds since `since`, from the first, changing once a second on the second.
  function count(line, since) {
    clearTimeout(line.tick);
    if (since === null) return;
    const elapsed = Date.now() - since;
    const seconds = Math.floor(elapsed / 1000);
    facts(line, seconds < 1 ? null : `${seconds}s`);
    line.tick = setTimeout(() => line.isConnected && count(line, since), 1000 - elapsed % 1000);
  }

  // The last message has Ask Again, as the app allows it, and its actions showing.
  function markLast() {
    for (const element of messages.querySelectorAll('.last')) element.classList.remove('last');
    const last = messages.lastElementChild;
    last?.classList.add('last');
    const again = last?.querySelector('.again');
    if (again) {
      again.disabled = !askAgain.enabled;
      again.title = askAgain.help;
    }
  }

  // The whole conversation: another opens at its latest message, and a message sent scrolls there.
  function show(page) {
    const opened = page.id !== conversation;
    if (opened) {
      views.clear();
      expanded.clear();
    }
    const added = !opened && page.messages.at(-1)?.id !== messages.lastElementChild?.dataset.id;
    conversation = page.id;
    askAgain = page.askAgain;
    const elements = page.messages.map(draw);
    const ids = new Set(page.messages.map(message => message.id));
    for (const id of views.keys()) if (!ids.has(id)) views.delete(id);
    if (elements.length !== messages.children.length || elements.some((element, index) => messages.children[index] !== element)) {
      messages.replaceChildren(...elements);
    }
    markLast();
    if (opened || added) {
      stopGliding();
      following = true;
      toBottom();
    }
  }

  // The reply that streams, alone.
  function update(message) {
    if (views.has(message.id)) draw(message);
  }

  messages.addEventListener('click', event => {
    const button = event.target.closest('button');
    const view = button && views.get(button.closest('article').dataset.id);
    if (!view) return;
    if (button.classList.contains('copy-reply')) copy(button, view.message.text);
    else if (button.classList.contains('copy-code')) copy(button, button.closest('figure').querySelector('code').textContent);
    else if (button.classList.contains('again')) post({askAgain: true});
    else if (button.classList.contains('toggle')) {
      expanded.set(view.message.id, !button.closest('.line').classList.contains('open'));
      drawLine(view.element, view.message);
    }
  });

  // Copies, and says so: the button shows a check for 1.5 s, which another copy holds longer.
  function copy(button, text) {
    post({copy: text});
    button.classList.add('copied');
    clearTimeout(button.copied);
    button.copied = setTimeout(() => button.classList.remove('copied'), 1500);
  }

  // MARK: Following the latest

  // The conversation follows its latest message while the reader is at the bottom, until they
  // scroll up, and again once they scroll back down; nothing moves it while they scroll.
  let following = true;
  // The glide to the latest message under way, as an animation frame.
  let gliding = null;
  // When the reader last scrolled by wheel, trackpad or keys, and whether the pointer is down, as
  // while selecting.
  let scrolled = -Infinity;
  let pressed = false;

  const bottom = () => scroller.scrollHeight - scroller.clientHeight;
  const atBottom = () => bottom() - scroller.scrollTop <= 40;
  const readerScrolls = () => pressed || performance.now() - scrolled < 200;
  const toBottom = () => { scroller.scrollTop = bottom(); };

  function follow() {
    if (following && gliding === null && !readerScrolls()) toBottom();
  }

  function scrolling() {
    scrolled = performance.now();
    stopGliding();
  }

  addEventListener('wheel', scrolling, {passive: true});
  addEventListener('keydown', scrolling, {passive: true});
  addEventListener('pointerdown', () => {
    pressed = true;
    stopGliding();
  }, {passive: true});
  for (const type of ['pointerup', 'pointercancel']) addEventListener(type, () => { pressed = false; }, {passive: true});
  addEventListener('scroll', () => {
    if (gliding === null) following = atBottom();
    showLatest();
  }, {passive: true});
  // A reply that grows, the thinking opening, the window resized.
  new ResizeObserver(follow).observe(messages);
  addEventListener('resize', follow);

  function showLatest() {
    latest.classList.toggle('shown', !following);
    latest.setAttribute('aria-hidden', following);
    latest.tabIndex = following ? -1 : 0;
  }

  // Back to the latest message, gliding as Motion.glide does: a smooth spring over 0.4 s, to
  // wherever the bottom is as it goes.
  latest.addEventListener('click', () => {
    following = true;
    showLatest();
    if (reduced.matches) return toBottom();
    const start = scroller.scrollTop;
    const began = performance.now();
    const spring = t => 1 - (1 + 7.4 * t) * Math.exp(-7.4 * t);
    const step = now => {
      const t = Math.min((now - began) / 400, 1);
      scroller.scrollTop = start + (bottom() - start) * spring(t) / spring(1);
      gliding = t < 1 ? requestAnimationFrame(step) : null;
    };
    gliding = requestAnimationFrame(step);
  });

  function stopGliding() {
    if (gliding === null) return;
    cancelAnimationFrame(gliding);
    gliding = null;
  }

  // MARK: Setup

  // The SF Symbols the page draws, which it can't name: masks, colored as text is, but the warning
  // sign, in its own colors.
  function setup({glyphs}) {
    const style = document.documentElement.style;
    for (const [name, glyph] of Object.entries(glyphs)) {
      style.setProperty(`--${name}`, `url("${glyph.url}")`);
      style.setProperty(`--${name}-width`, `${glyph.width}px`);
      style.setProperty(`--${name}-height`, `${glyph.height}px`);
    }
  }

  return {setup, show, update};
})();
