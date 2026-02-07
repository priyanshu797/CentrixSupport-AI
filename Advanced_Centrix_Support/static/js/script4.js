// Form handling and validation
document.addEventListener('DOMContentLoaded', function() {
    const contactForm = document.getElementById('contactForm');
    const contactContainer = document.querySelector('.contact-container');
  
    // Add fade-in animation on page load
    contactContainer.style.opacity = '0';
    contactContainer.style.transform = 'translateY(20px)';
    contactContainer.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    
    setTimeout(() => {
      contactContainer.style.opacity = '1';
      contactContainer.style.transform = 'translateY(0)';
    }, 100);
  
    // Form submission handler
    contactForm.addEventListener('submit', function(e) {
      // Optional: Add client-side validation before submission
      const name = contactForm.querySelector('input[name="name"]').value.trim();
      const email = contactForm.querySelector('input[name="email"]').value.trim();
      const message = contactForm.querySelector('textarea[name="message"]').value.trim();
  
      if (!name || !email || !message) {
        e.preventDefault();
        showMessage('Please fill in all fields.', 'error');
        return;
      }
  
      // Email validation
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(email)) {
        e.preventDefault();
        showMessage('Please enter a valid email address.', 'error');
        return;
      }
  
      // Disable submit button to prevent double submission
      const submitButton = contactForm.querySelector('button[type="submit"]');
      submitButton.disabled = true;
      submitButton.textContent = 'Sending...';
      
      // Form will submit normally to Flask backend
      // If you want AJAX submission, uncomment below and prevent default
      /*
      e.preventDefault();
      
      fetch('/send_email', {
        method: 'POST',
        body: new FormData(contactForm)
      })
      .then(response => response.json())
      .then(data => {
        if (data.success) {
          showMessage('Message sent successfully!', 'success');
          contactForm.reset();
        } else {
          showMessage('Failed to send message. Please try again.', 'error');
        }
        submitButton.disabled = false;
        submitButton.textContent = 'Send';
      })
      .catch(error => {
        showMessage('An error occurred. Please try again.', 'error');
        submitButton.disabled = false;
        submitButton.textContent = 'Send';
      });
      */
    });
  
    // Show message function
    function showMessage(text, type) {
      // Remove existing messages
      const existingMessages = contactContainer.querySelectorAll('.success-message, .error-message');
      existingMessages.forEach(msg => msg.remove());
  
      // Create new message
      const messageDiv = document.createElement('div');
      messageDiv.className = type === 'success' ? 'success-message show' : 'error-message show';
      messageDiv.textContent = text;
      
      contactContainer.appendChild(messageDiv);
  
      // Auto-hide after 5 seconds
      setTimeout(() => {
        messageDiv.classList.remove('show');
        setTimeout(() => messageDiv.remove(), 300);
      }, 5000);
    }
  
    // Add character counter for textarea (optional enhancement)
    const textarea = contactForm.querySelector('textarea[name="message"]');
    const maxLength = 500;
    
    textarea.addEventListener('input', function() {
      const remaining = maxLength - this.value.length;
      
      // Only show counter if approaching limit
      if (remaining < 100) {
        let counter = contactContainer.querySelector('.char-counter');
        if (!counter) {
          counter = document.createElement('div');
          counter.className = 'char-counter';
          counter.style.cssText = 'font-size: 0.85rem; color: #666; text-align: right; margin-top: -10px;';
          textarea.parentNode.insertBefore(counter, textarea.nextSibling);
        }
        counter.textContent = `${remaining} characters remaining`;
        counter.style.color = remaining < 50 ? '#d32f2f' : '#666';
      }
    });
  });