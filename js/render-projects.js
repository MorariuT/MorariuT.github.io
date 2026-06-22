// Render functions for project cards
function createProjectCard(project) {
  const card = document.createElement('div');
  card.className = `project-card project-${project.size}`;
  
  let content = `<h3><a href="${project.url || '#'}">${project.title}</a></h3>`;
  content += `<p>${project.description}</p>`;
  
  // Add media if present
  if (project.media) {
    if (project.media.type === 'image') {
      content += `<img src="${project.media.src}" alt="${project.media.alt}" class="project-media">`;
    } else if (project.media.type === 'video') {
      content += `<video controls class="project-media">
        <source src="${project.media.src}" type="video/mp4">
        Your browser does not support the video tag.
      </video>`;
    }
  }
  
  // Add links
  if (project.links) {
    content += '<div class="project-links">';
    project.links.forEach(link => {
      content += `<a href="${link.url}" class="project-link">${link.text}</a>`;
    });
    content += '</div>';
  }
  
  // Add sub-items if present
  if (project.subItems) {
    content += '<ul class="sub-items">';
    project.subItems.forEach(item => {
      content += `<li><a href="${item.url}">${item.text}</a></li>`;
    });
    content += '</ul>';
  }
  
  card.innerHTML = content;
  return card;
}

function renderProjectsSection(containerId, projects, title) {
  const container = document.getElementById(containerId);
  if (!container) return;
  
  const section = document.createElement('div');
  section.className = 'projects-section';
  
  if (title) {
    const heading = document.createElement('h1');
    heading.className = 'section-title';
    heading.innerHTML = title;
    section.appendChild(heading);
  }
  
  const grid = document.createElement('div');
  grid.className = 'projects-grid';
  
  projects.forEach(project => {
    grid.appendChild(createProjectCard(project));
  });
  
  section.appendChild(grid);
  container.appendChild(section);
}

function renderProjectsByCategory(containerId, categoriesData) {
  const container = document.getElementById(containerId);
  if (!container) return;
  
  Object.entries(categoriesData).forEach(([categoryKey, projects]) => {
    const categoryName = categoryKey
      .replace(/([A-Z])/g, ' $1')
      .trim()
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
    
    renderProjectsSection(containerId, projects, `<a href="#">${categoryName}</a>`);
  });
}
