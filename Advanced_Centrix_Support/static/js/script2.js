// Handle continue button click
function handleContinue(button) {
  const checkbox = document.getElementById("agree");

  if (!checkbox.checked) {
    alert("⚠️ You must agree to the disclaimer to continue.");
    return;
  }

  // ✅ Read Flask-generated URL from HTML
  const nextUrl = button.getAttribute("data-next-url");
  window.location.href = nextUrl;
}

// Optional: Enable continue on Enter key
document.addEventListener('DOMContentLoaded', function () {
  const checkbox = document.getElementById('agree');
  const continueBtn = document.querySelector('.btn-continue');

  checkbox.addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
      checkbox.checked = !checkbox.checked;
    }
  });

  continueBtn.addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
      handleContinue(continueBtn);
    }
  });
});
