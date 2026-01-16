// Resources page interactions and enhancements
document.addEventListener('DOMContentLoaded', function() {
    // Add fade-in animation for page intro
    const pageIntro = document.querySelector('.page-intro');
    if (pageIntro) {
      pageIntro.style.opacity = '0';
      pageIntro.style.transform = 'translateY(-10px)';
      pageIntro.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
      
      setTimeout(() => {
        pageIntro.style.opacity = '1';
        pageIntro.style.transform = 'translateY(0)';
      }, 100);
    }
  
    // Add staggered fade-in animation for resource cards
    const resourceCards = document.querySelectorAll('.resource-card');
    
    resourceCards.forEach((card, index) => {
      card.style.opacity = '0';
      card.style.transform = 'translateY(20px)';
      card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
      
      setTimeout(() => {
        card.style.opacity = '1';
        card.style.transform = 'translateY(0)';
      }, 200 + (index * 100));
    });
  
    // Track external link clicks (analytics ready)
    const resourceLinks = document.querySelectorAll('.resource-link');
    
    resourceLinks.forEach(link => {
      link.addEventListener('click', function(e) {
        const resourceTitle = this.closest('.resource-card').querySelector('.resource-title').textContent;
        const linkUrl = this.href;
        
        // Log the click (can be replaced with actual analytics)
        console.log(`Resource clicked: ${resourceTitle} - ${linkUrl}`);
        
        // Optional: Add visual feedback
        this.style.opacity = '0.7';
        setTimeout(() => {
          this.style.opacity = '1';
        }, 200);
      });
    });
  
    // Add keyboard navigation
    document.addEventListener('keydown', function(e) {
      // Press 'B' to go back to chat
      if ((e.key === 'b' || e.key === 'B') && !e.ctrlKey && !e.metaKey) {
        const backButton = document.querySelector('.back-button');
        if (backButton) {
          window.location.href = backButton.href;
        }
      }
  
      // Press number keys 1-4 to visit resources
      if (e.key >= '1' && e.key <= '4' && !e.ctrlKey && !e.metaKey) {
        const index = parseInt(e.key) - 1;
        const card = resourceCards[index];
        if (card) {
          const link = card.querySelector('.resource-link');
          if (link) {
            window.open(link.href, '_blank');
          }
        }
      }
    });
  
    // Add intersection observer for cards (alternative animation)
    const observerOptions = {
      threshold: 0.1,
      rootMargin: '0px 0px -50px 0px'
    };
  
    const observer = new IntersectionObserver(function(entries) {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          // Add a subtle pulse effect when card becomes visible
          entry.target.style.animation = 'pulse 0.5s ease-in-out';
        }
      });
    }, observerOptions);
  
    // Observe each resource card
    resourceCards.forEach(card => {
      observer.observe(card);
    });
  
    // Add pulse animation
    const style = document.createElement('style');
    style.textContent = `
      @keyframes pulse {
        0%, 100% {
          transform: scale(1);
        }
        50% {
          transform: scale(1.02);
        }
      }
    `;
    document.head.appendChild(style);
  
    // Add copy link functionality (right-click or long press)
    resourceCards.forEach(card => {
      const link = card.querySelector('.resource-link');
      
      card.addEventListener('contextmenu', function(e) {
        e.preventDefault();
        
        // Copy link to clipboard
        const url = link.href;
        navigator.clipboard.writeText(url).then(() => {
          showNotification('Link copied to clipboard!');
        }).catch(err => {
          console.error('Failed to copy link:', err);
        });
      });
    });
  
    // Show notification function
    function showNotification(message) {
      const notification = document.createElement('div');
      notification.textContent = message;
      notification.style.cssText = `
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        background-color: #0f766e;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        z-index: 1000;
        animation: slideIn 0.3s ease;
      `;
      
      document.body.appendChild(notification);
      
      // Remove after 3 seconds
      setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
      }, 3000);
    }
  
    // Add slide animations for notification
    const notificationStyle = document.createElement('style');
    notificationStyle.textContent = `
      @keyframes slideIn {
        from {
          transform: translateX(100%);
          opacity: 0;
        }
        to {
          transform: translateX(0);
          opacity: 1;
        }
      }
      @keyframes slideOut {
        from {
          transform: translateX(0);
          opacity: 1;
        }
        to {
          transform: translateX(100%);
          opacity: 0;
        }
      }
    `;
    document.head.appendChild(notificationStyle);
  
    // Add hover effect information on first visit
    let hasSeenTip = sessionStorage.getItem('resourcesTipSeen');
    
    if (!hasSeenTip) {
      setTimeout(() => {
        showNotification('💡 Tip: Right-click any card to copy the link!');
        sessionStorage.setItem('resourcesTipSeen', 'true');
      }, 2000);
    }
  });