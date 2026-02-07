// Handle back button click
document.addEventListener('DOMContentLoaded', function() {
    const backBtn = document.getElementById('backBtn');
    
    backBtn.addEventListener('click', () => {
      // Navigate back to chat page
      // Replace with your actual Flask route
      window.location.href = "{{ url_for('index') }}";
    });
  
    // Optional: Add keyboard accessibility - pressing Escape also goes back
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape') {
        window.location.href = "{{ url_for('index') }}";
      }
    });
  
    // Optional: Add animation on page load
    const helpContainer = document.querySelector('.help-container');
    helpContainer.style.opacity = '0';
    helpContainer.style.transform = 'translateY(20px)';
    helpContainer.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    
    setTimeout(() => {
      helpContainer.style.opacity = '1';
      helpContainer.style.transform = 'translateY(0)';
    }, 100);
  });