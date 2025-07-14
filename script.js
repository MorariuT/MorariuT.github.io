const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

canvas.width = 400;
canvas.height = 200;

// Bird
let bird = {
  x: 80,
  y: 100,
  width: 10,
  height: 10,
  velocity: 0,
  gravity: 0.2,
  lift: -5
};

// Pipes
let pipes = [];
let pipeGap = 90;
let pipeWidth = 40;
let pipeSpeed = 3;
let frame = 0;
let score = 0;

function drawBird() {
  ctx.fillStyle = "yellow";
  ctx.fillRect(bird.x, bird.y, bird.width, bird.height);
}

function drawPipes() {
  ctx.fillStyle = "green";
  for (let p of pipes) {
    ctx.fillRect(p.x, 0, pipeWidth, p.top);
    ctx.fillRect(p.x, p.bottom, pipeWidth, canvas.height - p.bottom);
  }
}

function updatePipes() {
  if (frame % 90 === 0) {
    let top = Math.random() * (canvas.height / 2);
    pipes.push({
      x: canvas.width,
      top: top,
      bottom: top + pipeGap
    });
  }

  for (let i = pipes.length - 1; i >= 0; i--) {
    pipes[i].x -= pipeSpeed;

    // Collision
    if (
      bird.x < pipes[i].x + pipeWidth &&
      bird.x + bird.width > pipes[i].x &&
      (bird.y < pipes[i].top || bird.y + bird.height > pipes[i].bottom)
    ) {
      resetGame();
      break;
    }

    // Passed pipe
    if (pipes[i].x + pipeWidth < bird.x && !pipes[i].scored) {
      score++;
      pipes[i].scored = true;
    }

    // Remove off-screen pipes
    if (pipes[i].x + pipeWidth < 0) {
      pipes.splice(i, 1);
    }
  }
}

function drawScore() {
  ctx.fillStyle = "white";
  ctx.font = "10px Arial";
  ctx.fillText(`Score: ${score}`, 5, 10);
}

function resetGame() {
  bird.y = 100;
  bird.velocity = 0;
  pipes = [];
  score = 0;
  frame = 0;
}

function update() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  bird.velocity += bird.gravity;
  bird.y += bird.velocity;

  if (bird.y + bird.height > canvas.height || bird.y < 0) {
    resetGame();
  }

  drawBird();
  updatePipes();
  drawPipes();
  drawScore();

  frame++;
  requestAnimationFrame(update);
}

document.addEventListener("keydown", function (e) {
  if (e.code === "Space") {
    bird.velocity = bird.lift;
  }
});

update();
