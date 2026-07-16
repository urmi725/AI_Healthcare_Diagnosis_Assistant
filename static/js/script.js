document.addEventListener("DOMContentLoaded", function () {
  // Symptom search filter
  var searchBox = document.getElementById("symptomSearch");
  var chipItems = document.querySelectorAll(".chip-item");

  if (searchBox) {
    searchBox.addEventListener("input", function () {
      var term = this.value.toLowerCase();
      chipItems.forEach(function (item) {
        var label = item.querySelector("label").textContent.toLowerCase();
        item.classList.toggle("hidden", term.length > 0 && !label.includes(term));
      });
    });
  }

  // Live "N selected" counter
  var counter = document.getElementById("symptomCounter");
  var checkboxes = document.querySelectorAll('.chip-item input[type="checkbox"]');

  function updateCounter() {
    if (!counter) return;
    var count = document.querySelectorAll('.chip-item input[type="checkbox"]:checked').length;
    counter.innerHTML = count === 0
      ? "No symptoms selected yet"
      : "<strong>" + count + "</strong> symptom" + (count === 1 ? "" : "s") + " selected";
  }

  checkboxes.forEach(function (cb) {
    cb.addEventListener("change", updateCounter);
  });
  updateCounter();
});
