// Smooth scroll for internal links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function(e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      target.scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
      });
    }
  });
});

// Add fade-in animation for feature cards on page load
document.addEventListener('DOMContentLoaded', function() {
  const featureCards = document.querySelectorAll('.feature-card');
  
  // Add visible class to cards with staggered timing
  featureCards.forEach((card, index) => {
    setTimeout(() => {
      card.classList.add('visible');
    }, index * 150); // 150ms delay between each card
  });

  // Optional: Add intersection observer for cards that scroll into view
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  }, observerOptions);

  // Observe all feature cards
  featureCards.forEach(card => {
    observer.observe(card);
  });
});

// Add hover effect enhancement
document.addEventListener('DOMContentLoaded', function() {
  const cards = document.querySelectorAll('.feature-card');
  
  cards.forEach(card => {
    card.addEventListener('mouseenter', function() {
      this.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
    });
  });
});

// Optional: Log when page is fully loaded
window.addEventListener('load', function() {
  console.log('Learn More page loaded successfully');
});