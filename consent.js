/* BrickWorld cookie consent + Google Analytics.
   Analytics loads only after the visitor clicks "Accept".
   The choice is saved in this browser; "Cookie settings" in the footer asks again. */
(function () {
  var GA_ID = 'G-L6B4CL5WYQ';
  var KEY = 'bw-consent';

  function getChoice() { try { return localStorage.getItem(KEY); } catch (e) { return null; } }
  function setChoice(v) { try { localStorage.setItem(KEY, v); } catch (e) {} }

  function loadAnalytics() {
    if (window.__bwGA) return;
    window.__bwGA = true;
    window.dataLayer = window.dataLayer || [];
    window.gtag = function () { dataLayer.push(arguments); };
    gtag('js', new Date());
    gtag('config', GA_ID);
    var s = document.createElement('script');
    s.async = true;
    s.src = 'https://www.googletagmanager.com/gtag/js?id=' + GA_ID;
    document.head.appendChild(s);
  }

  function addStyles() {
    if (document.getElementById('bw-consent-style')) return;
    var css =
      '#bw-consent{position:fixed;left:16px;right:16px;bottom:16px;z-index:1000;max-width:560px;margin:0 auto;' +
      'background:#170a01;border:2px solid #3a1d06;border-radius:4px;padding:20px;color:#f0b27a;' +
      "font-family:'IBM Plex Mono',ui-monospace,monospace;font-size:15px;line-height:1.5;box-shadow:0 8px 24px rgba(0,0,0,.5)}" +
      '#bw-consent p{margin:0 0 14px}#bw-consent a{color:#ff8a1f}' +
      '#bw-consent .bw-row{display:flex;flex-wrap:wrap;gap:12px}' +
      '#bw-consent button{flex:1 1 140px;min-height:44px;padding:0 20px;font:inherit;font-weight:600;text-transform:uppercase;' +
      'letter-spacing:.08em;border-radius:2px;cursor:pointer;border:1px solid #ff8a1f}' +
      '#bw-consent .bw-yes{background:#ff8a1f;color:#120800}#bw-consent .bw-no{background:none;color:#ff8a1f}' +
      '#bw-consent button:focus-visible{outline:2px solid #ffb066;outline-offset:2px}';
    var st = document.createElement('style');
    st.id = 'bw-consent-style';
    st.textContent = css;
    document.head.appendChild(st);
  }

  function showBanner() {
    if (document.getElementById('bw-consent')) return;
    addStyles();
    var box = document.createElement('div');
    box.id = 'bw-consent';
    box.setAttribute('role', 'dialog');
    box.setAttribute('aria-label', 'Cookies');
    box.innerHTML =
      '<p>I use Google Analytics to see how many people visit the site and which pages they like. ' +
      'It only runs if you say yes. <a href="privacy.html">Privacy</a></p>' +
      '<div class="bw-row"><button type="button" class="bw-yes">Accept</button>' +
      '<button type="button" class="bw-no">No thanks</button></div>';
    document.body.appendChild(box);
    box.querySelector('.bw-yes').addEventListener('click', function () {
      setChoice('granted'); box.remove(); loadAnalytics();
    });
    box.querySelector('.bw-no').addEventListener('click', function () {
      setChoice('denied'); box.remove();
    });
  }

  // "Cookie settings" links: <a href="#" data-cookie-settings>
  document.addEventListener('click', function (e) {
    var a = e.target.closest && e.target.closest('[data-cookie-settings]');
    if (!a) return;
    e.preventDefault();
    if (getChoice() === 'granted') {
      // Turning analytics off needs a reload so the script is gone.
      setChoice('denied');
      document.cookie.split(';').forEach(function (c) {
        var n = c.split('=')[0].trim();
        if (n.indexOf('_ga') === 0) {
          document.cookie = n + '=; Max-Age=0; path=/; domain=.' + location.hostname.replace(/^www\./, '');
          document.cookie = n + '=; Max-Age=0; path=/';
        }
      });
      try { localStorage.removeItem(KEY); } catch (e2) {}
      location.reload();
      return;
    }
    try { localStorage.removeItem(KEY); } catch (e3) {}
    showBanner();
  });

  function start() {
    var c = getChoice();
    if (c === 'granted') loadAnalytics();
    else if (c !== 'denied') showBanner();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
