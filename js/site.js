// Shared navigation and clock for every page. No dependencies or build step.
const clock = document.querySelector('#clock');
function updateClock() {
  const now = new Date();
  clock.dateTime = now.toISOString();
  clock.textContent = new Intl.DateTimeFormat('en-GB', {
    day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit'
  }).format(now);
}
updateClock();
setInterval(updateClock, 30_000);
const workspaces = [...document.querySelectorAll('.bar nav a')];
function syncWorkspace() {
  if (document.body.dataset.workspace !== 'home') return;
  const active = location.hash === '#projects' ? 1 : 0;
  workspaces.forEach((link, i) => {
    if (i === active) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
}
window.addEventListener('hashchange', syncWorkspace);
syncWorkspace();
document.addEventListener('keydown', event => {
  if (event.target.closest('input, textarea, select, button, [contenteditable]')) return;
  if (event.altKey && !event.ctrlKey && !event.metaKey && ['1', '2', '3', '4'].includes(event.key)) {
    event.preventDefault();
    workspaces[Number(event.key) - 1].click();
  }
});
const outlines = [...document.querySelectorAll('.outline a')];
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    const visible = entries.filter(entry => entry.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
    if (!visible) return;
    outlines.forEach(link => {
      if (link.hash === `#${visible.target.id}`) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }, { rootMargin: '-65px 0px -40% 0px' });
  outlines.forEach(link => { const section = document.querySelector(link.hash); if (section) observer.observe(section); });
}
async function loadRepositories() {
  const container = document.querySelector('#repos');
  if (!container) return;
  try {
    const response = await fetch('https://api.github.com/users/MorariuT/repos?sort=created&per_page=3', {
      signal: AbortSignal.timeout(8000)
    });
    if (!response.ok) throw new Error(`GitHub returned ${response.status}`);
    const repos = await response.json();
    if (!Array.isArray(repos)) throw new Error('Unexpected response');
    const fragment = document.createDocumentFragment();
    for (const repo of repos) {
      const url = new URL(repo.html_url);
      if (url.origin !== 'https://github.com') continue;
      const link = document.createElement('a');
      link.className = 'repo';
      link.href = url.href;
      const name = document.createElement('span');
      name.textContent = `${repo.name} ↗`;
      const description = document.createElement('p');
      description.textContent = repo.description || 'Explore this repository on GitHub.';
      link.append(name, description);
      fragment.append(link);
    }
    if (!fragment.childElementCount) throw new Error('No repositories');
    container.replaceChildren(fragment);
  } catch {
    const message = document.createElement('p');
    message.textContent = 'The feed is unavailable. You can still browse all repositories on GitHub.';
    container.replaceChildren(message);
  }
}
loadRepositories();
