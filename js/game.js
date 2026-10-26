
        var canvas = document.getElementById("galtonCanvas");
        var ctx = canvas.getContext("2d");

        var width = canvas.width;
        var height = canvas.height;
        var pinRows = 5;
        var ballRadius = 5;
        var gravity = 0.5;
        var friction = 0.98;
        var num_balls = 150;
        let pins = [];
        var binHeights = new Array(pinRows + 1).fill(0);
        var balls = [];
        var binWidth = width / binHeights.length;

        var slider = document.getElementById("noRows");
        var output = document.getElementById("sliderRows");

        output.innerHTML = 5;

        var slider_ball = document.getElementById("noBall");
        var output_ball = document.getElementById("sliderBall");

        output_ball.innerHTML = 5;

        var slider_no_ball = document.getElementById("NumBall");
        var output_no_ball = document.getElementById("sliderNumBall");

        output_no_ball.innerHTML = num_balls;

        slider.oninput = function()
        {
            // ctx.clearRect(0, 0, canvas.width, canvas.height);
            pins = [];
            output.innerHTML = this.value;

            // pinRows = this.value;
            pinRows = parseInt(this.value);
            binHeights = new Array(pinRows + 1).fill(0);

            binWidth = width / binHeights.length;
        }

        slider_ball.oninput = function()
        {
            // ctx.clearRect(0, 0, canvas.width, canvas.height);
            pins = [];
            output_ball.innerHTML = this.value;

            // pinRows = this.value;
            ballRadius = parseInt(this.value);
        }

        slider_no_ball.oninput = function()
        {
            // ctx.clearRect(0, 0, canvas.width, canvas.height);
            output_no_ball.innerHTML = this.value;

            // pinRows = this.value;
            num_balls = parseInt(this.value);
        }




        // Generate pin positions
        function generatePins()
        {
            pins = [];
            const spacingX = width / pinRows;
            const spacingY = height / (pinRows + 2);
            for (let row = 0; row < pinRows; row++)
            {
                for (let col = 0; col <= row; col++)
                {
                    const x = spacingX * (col + 0.5 * (pinRows - row));
                    const y = spacingY * (row + 1);
                    pins.push({ x, y });
                }
            }
        }

        function drawPin(x, y)
        {
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fillStyle = "#90ee90";
            ctx.fill();
        }

        function drawBall(ball)
        {
            ctx.beginPath();
            ctx.arc(ball.x, ball.y, ballRadius, 0, Math.PI * 2);
            ctx.fillStyle = "#00ff9c";
            ctx.fill();
        }

        function drawBins()
        {
            for (let i = 0; i < binHeights.length; i++)
            {
                ctx.fillStyle = "#90ee90";
                ctx.fillRect(i * binWidth, height - binHeights[i], binWidth - 2, binHeights[i]);
            }
        }

        // Init i minge noua
        function createBall()
        {
            return { //json
                x: width / 2,
                y: 0,
                vx: 0,
                vy: 2,
                active: true,
            };
        }

        function updateBall(ball) {
            if (!ball.active) return;

            ball.vy += gravity;

            // Update pos
            ball.x += ball.vx;
            ball.y += ball.vy;

            // Verif fiecare pin pt coliziune
            for (const pin of pins) {
                const dx = ball.x - pin.x;
                const dy = ball.y - pin.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < ballRadius + 3)
                {
                    // Detectam o coliziune
                    // idk de ce merge codu dar merge.
                    const angle = Math.atan2(dy, dx);
                    ball.vx = Math.cos(angle) * 2 + (Math.random() - 0.5);
                    ball.vy = Math.sin(angle) * 2;
                    break;
                }
            }

            ball.vx *= friction;

            if (ball.y >= height - ballRadius)
            {
                const binIndex = Math.floor(ball.x / binWidth);
                if (binIndex >= 0 && binIndex < binHeights.length)
                {
                    binHeights[binIndex]++;
                }
                ball.active = false;
            }
        }

        // Fucntie scirsa de chatgpt :))
        function handlePinClick(event)
        {
            const rect = canvas.getBoundingClientRect();
            const mouseX = (event.clientX - rect.left) * canvas.width / rect.width;
            const mouseY = (event.clientY - rect.top) * canvas.height / rect.height;

            for (let i = 0; i < pins.length; i++)
            {
                const pin = pins[i];
                const dx = mouseX - pin.x;
                const dy = mouseY - pin.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < 5)
                {
                    pins.splice(i, 1);
                    drawScene();
                    break;
                }
            }
        }

        function drawScene()
        {
            ctx.clearRect(0, 0, width, height);

            pins.forEach(pin => drawPin(pin.x, pin.y));
            drawBins();

            balls.forEach(ball =>
            {
                updateBall(ball);
                if (ball.active)
                {
                    drawBall(ball);
                }
            });
        }


let spawnInterval;
let animationFrame;
function animate() {
    drawScene();
    if (balls.some(ball => ball.active) || spawnInterval) {
        animationFrame = requestAnimationFrame(animate);
    } else {
        animationFrame = undefined;
        document.getElementById('startSimulation').textContent = 'Start simulation';
    }
}
function startSimulation() {
    clearInterval(spawnInterval);
    cancelAnimationFrame(animationFrame);
    balls = [];
    binHeights = new Array(pinRows + 1).fill(0);
    binWidth = width / binHeights.length;
    generatePins();
    const total = num_balls;
    spawnInterval = setInterval(() => {
        balls.push(createBall());
        if (balls.length >= total) {
            clearInterval(spawnInterval);
            spawnInterval = undefined;
        }
    }, 300);
    document.getElementById('startSimulation').textContent = 'Restart simulation';
    animate();
}
canvas.addEventListener('click', handlePinClick);
document.getElementById('startSimulation').addEventListener('click', startSimulation);
generatePins();
drawScene();
