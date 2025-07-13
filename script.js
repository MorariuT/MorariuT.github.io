    const items = document.querySelectorAll('.menu-item');
    let current = 0;

    function updateSelection() {
      items.forEach((item, index) => {
        item.classList.toggle('selected', index === current);
      });
      document.getElementById("selected-option").textContent = items[current].textContent;
    }

    document.addEventListener('keydown', e => {
      if (e.key === 'ArrowDown') {
        current = (current + 1) % items.length;
        updateSelection();
      } else if (e.key === 'ArrowUp') {
        current = (current - 1 + items.length) % items.length;
        updateSelection();
      } else if (e.key === 'Enter') {
        counter = 0;
        
        // create new span
        const newSpan = document.createElement('span');
        newSpan.textContent = counter;
        
        // Insert the new span before the focused and time spans,
        // or just append it to the end if you want
        // Let's insert before the 'focused' span for demo:
        
        const focusedSpan = document.getElementById('focused');
        statusBar.insertBefore(newSpan, focusedSpan);
        
        // If you want to append at the end, just do:
        // statusBar.appendChild(newSpan);
          updateSelection();

      }
    });

    updateSelection();