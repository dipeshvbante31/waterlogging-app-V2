(function () {
  "use strict";
  var form = document.getElementById("manual-form");
  if (!form) return;
  var chain = ["precip_current_mm", "precip_past_3h_mm", "precip_past_24h_mm", "precip_past_72h_mm"];
  function el(name) { return document.getElementById("f-" + name); }
  function num(input) { return input && input.value !== "" ? Number(input.value) : null; }

  // Client-side checks mirror the server's rules; the server validates again regardless.
  function validate() {
    chain.forEach(function (n) { el(n).setCustomValidity(""); });
    el("precip_next_6h_mm").setCustomValidity("");
    for (var i = 1; i < chain.length; i++) {
      var shorter = num(el(chain[i - 1])), longer = num(el(chain[i]));
      if (shorter !== null && longer !== null && longer + 1e-9 < shorter) {
        el(chain[i]).setCustomValidity("A longer window includes the shorter one, so this must be at least " + shorter + " mm.");
      }
    }
    if (el("precip_current_mm").value === "" && el("precip_next_6h_mm").value === "") {
      el("precip_current_mm").setCustomValidity("Enter current rainfall or the 6-hour forecast rainfall.");
    }
  }

  form.addEventListener("input", validate);
  form.addEventListener("submit", function (event) {
    validate();
    if (!form.checkValidity()) {
      event.preventDefault();
      form.reportValidity();
    }
  });

  var result = document.getElementById("result");
  if (result) {
    result.focus({ preventScroll: true });
    result.scrollIntoView({ behavior: "smooth", block: "start" });
  } else {
    var bad = form.querySelector("[aria-invalid='true']");
    if (bad) bad.focus();
  }
})();
