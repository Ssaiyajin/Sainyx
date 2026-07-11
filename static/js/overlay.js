function spawnParticles() {
    const c = document.getElementById('spark-container');
    c.innerHTML = '';
    for (let i = 0; i < 30; i++) {
        const s = document.createElement('div');
        s.className = 'spark';
        s.style.left = (15+Math.random()*70)+'%';
        s.style.top  = (10+Math.random()*80)+'%';
        s.style.width = s.style.height = (1+Math.random()*3)+'px';
        s.style.background = Math.random()>0.5 ? '#ffffff' : '#cc88ff';
        s.style.setProperty('--dx', (Math.random()-0.5)*150+'px');
        s.style.setProperty('--dy', -(Math.random()*150+30)+'px');
        s.style.animationDelay    = (Math.random()*0.5)+'s';
        s.style.animationDuration = (0.3+Math.random()*0.4)+'s';
        c.appendChild(s);
    }
    for (let i = 0; i < 12; i++) {
        const p = document.createElement('div');
        p.className = 'particle';
        const sz = 4+Math.random()*10;
        p.style.width = p.style.height = sz+'px';
        p.style.left = (10+Math.random()*80)+'%';
        p.style.top  = (10+Math.random()*80)+'%';
        p.style.setProperty('--sx', (Math.random()-0.5)*200+'px');
        p.style.setProperty('--sy', -(Math.random()*200)+'px');
        p.style.animationDelay    = (Math.random()*1)+'s';
        p.style.animationDuration = (1+Math.random()*1.5)+'s';
        c.appendChild(p);
    }
}

function startOverlay() {
    spawnParticles();
    const o = document.getElementById('overlay');
    o.style.transition = 'opacity 0.4s';
    o.classList.add('active');
    o.style.opacity = '1';
}

function stopOverlay() {
    const o = document.getElementById('overlay');
    o.style.transition = 'opacity 0.8s';
    o.style.opacity = '0';
    setTimeout(() => { o.classList.remove('active'); o.style.opacity = ''; }, 800);
}