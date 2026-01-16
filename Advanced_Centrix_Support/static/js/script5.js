// Page interactions and enhancements
document.addEventListener('DOMContentLoaded', function() {
    // Add smooth scroll for any anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
          target.scrollIntoView({ behavior: 'smooth' });
        }
      });
    });
  
    // Add fade-in animation for content sections
    const sections = document.querySelectorAll('.content-section');
    
    sections.forEach((section, index) => {
      section.style.opacity = '0';
      section.style.transform = 'translateY(20px)';
      section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
      
      setTimeout(() => {
        section.style.opacity = '1';
        section.style.transform = 'translateY(0)';
      }, 100 + (index * 150));
    });
  
    // Add fade-in animation for page header
    const pageHeader = document.querySelector('.page-header');
    pageHeader.style.opacity = '0';
    pageHeader.style.transform = 'translateY(-20px)';
    pageHeader.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    
    setTimeout(() => {
      pageHeader.style.opacity = '1';
      pageHeader.style.transform = 'translateY(0)';
    }, 50);
  
    // Highlight active nav link based on current page
    highlightActiveNavLink();
  
    // Add feature list item hover effect
    const featureItems = document.querySelectorAll('.feature-list li');
    featureItems.forEach(item => {
      item.style.transition = 'transform 0.2s ease, color 0.2s ease';
      item.style.cursor = 'default';
      
      item.addEventListener('mouseenter', function() {
        this.style.transform = 'translateX(5px)';
        this.style.color = '#0f766e';
      });
      
      item.addEventListener('mouseleave', function() {
        this.style.transform = 'translateX(0)';
        this.style.color = '#374151';
      });
    });
  });
  
  // Function to highlight active navigation link
  function highlightActiveNavLink() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
      const linkPath = new URL(link.href).pathname;
      if (linkPath === currentPath) {
        link.style.fontWeight = 'bold';
        link.style.textDecoration = 'underline';
      }
    });
  }
  
  // Add scroll-to-top button (optional enhancement)
  function createScrollToTopButton() {
    const scrollBtn = document.createElement('button');
    scrollBtn.innerHTML = '↑';
    scrollBtn.className = 'scroll-to-top';
    scrollBtn.style.cssText = `
      position: fixed;
      bottom: 2rem;
      right: 2rem;
      background-color: #0f766e;
      color: white;
      border: none;
      border-radius: 50%;
      width: 3rem;
      height: 3rem;
      font-size: 1.5rem;
      cursor: pointer;
      opacity: 0;
      transition: opacity 0.3s ease, transform 0.2s ease;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      z-index: 1000;
    `;
    
    document.body.appendChild(scrollBtn);
    
    // Show button when scrolled down
    window.addEventListener('scroll', () => {
      if (window.scrollY > 300) {
        scrollBtn.style.opacity = '1';
      } else {
        scrollBtn.style.opacity = '0';
      }
    });
    
    // Scroll to top on click
    scrollBtn.addEventListener('click', () => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    
    // Hover effect
    scrollBtn.addEventListener('mouseenter', () => {
      scrollBtn.style.transform = 'scale(1.1)';
    });
    
    scrollBtn.addEventListener('mouseleave', () => {
      scrollBtn.style.transform = 'scale(1)';
    });
  }
  
  // Initialize scroll-to-top button
  createScrollToTopButton();