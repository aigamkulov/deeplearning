async function runExercise(number){
 const btn=document.getElementById("runBtn"), output=document.getElementById("output"), status=document.getElementById("status");
 if(!btn)return;
 btn.disabled=true;btn.textContent="⏳ Ishlamoqda...";status.className="status running";status.textContent="Dastur bajarilmoqda...";output.textContent="";
 try{
  const r=await fetch(`/run/${number}`,{method:"POST"}); const data=await r.json();
  output.textContent=data.output||"(Natija chiqarilmadi.)";
  status.className=data.ok?"status success":"status error";
  status.textContent=data.ok?"✓ Muvaffaqiyatli bajarildi":`✕ Xatolik (kod: ${data.returncode??"—"})`;
 }catch(e){status.className="status error";status.textContent="✕ Server bilan aloqa xatosi";output.textContent=String(e)}
 finally{btn.disabled=false;btn.textContent="▶ RUN"}
}
function copyCode(){
 const el=document.querySelector("#code code"),btn=document.getElementById("copyBtn"); if(!el||!btn)return;
 navigator.clipboard.writeText(el.innerText).then(()=>{btn.textContent="✓ Nusxalandi";setTimeout(()=>btn.textContent="⧉ Kodni nusxalash",1400)})
}
function addFormulaDetail(){
 const box=document.getElementById("detailRows"); if(!box)return;
 const row=document.createElement("div"); row.className="detail-row";
 row.innerHTML=`<input name="detail_formula" placeholder="Formula"><input name="detail_explanation" placeholder="Izoh"><textarea name="detail_code" rows="2" placeholder="Python kodi"></textarea>`;
 box.appendChild(row);
}
function toggleSidebar(){document.getElementById("sidebar")?.classList.toggle("open")}
function toggleTheme(){
 document.body.classList.toggle("dark");
 localStorage.setItem("dl-theme",document.body.classList.contains("dark")?"dark":"light");
}
if(localStorage.getItem("dl-theme")==="dark")document.body.classList.add("dark");


async function runExample(key, button) {
  const card = button.closest('.example-runner-card');
  const status = card.querySelector('.example-status');
  const output = card.querySelector('.example-output');
  const results = card.querySelector('.example-results');

  button.disabled = true;
  button.textContent = '⏳ ISHLAYAPTI...';
  status.classList.remove('hidden');
  status.textContent = 'Python misoli ishga tushirilmoqda...';
  output.textContent = 'Natija kutilmoqda...';
  results.innerHTML = '';

  try {
    const response = await fetch('/run-example/' + key, { method: 'POST' });
    const data = await response.json();

    status.textContent = data.ok ? '✓ Muvaffaqiyatli bajarildi' : '✕ Xatolik';
    output.textContent = data.output || '(Natija chiqarilmadi.)';

    if (data.images && data.images.length) {
      const title = document.createElement('div');
      title.className = 'results-title';
      title.textContent = '🖼️ Dastur hosil qilgan rasmlar (' + data.images.length + ')';
      results.appendChild(title);

      const grid = document.createElement('div');
      grid.className = 'example-result-grid';

      data.images.forEach((src, index) => {
        const figure = document.createElement('figure');
        figure.className = 'example-result-item';

        const img = document.createElement('img');
        img.src = src + '?v=' + Date.now();
        img.alt = key.toUpperCase() + ' natija ' + (index + 1);
        img.loading = 'lazy';

        const caption = document.createElement('figcaption');
        caption.textContent = src.split('/').pop();

        figure.appendChild(img);
        figure.appendChild(caption);
        grid.appendChild(figure);
      });
      results.appendChild(grid);
    }
  } catch (error) {
    status.textContent = '✕ Server bilan bog‘lanishda xatolik';
    output.textContent = String(error);
  } finally {
    button.disabled = false;
    button.textContent = '▶ RUN ' + key.toUpperCase();
  }
}
