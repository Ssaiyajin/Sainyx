const BOOT_LINES = ['bl1','bl2','bl3','bl4','bl5','bl6'];

async function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function runBoot() {
    await sleep(300);
    document.getElementById('boot-logo').classList.add('show');
    await sleep(800);
    document.getElementById('boot-bar-wrap').style.opacity = '1';

    for (let i = 0; i < BOOT_LINES.length; i++) {
        document.getElementById(BOOT_LINES[i]).classList.add('show');
        document.getElementById('boot-bar').style.width = ((i+1)/BOOT_LINES.length*100)+'%';
        await sleep(350);
    }

    await sleep(400);
    document.getElementById('boot-ready').style.opacity = '1';
    await sleep(700);

    const boot = document.getElementById('boot');
    boot.style.transition = 'opacity 0.6s';
    boot.style.opacity = '0';
    document.getElementById('app').classList.add('show');
    await sleep(600);
    boot.style.display = 'none';

    showWelcome();
}

runBoot();