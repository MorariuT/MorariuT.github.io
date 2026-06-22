// Main projects data for the portfolio
const mainProjects = [
  {
    title: "2D SDL Raycaster",
    description: "A raycaster that projects every ray onto objects.",
    size: "medium",
    links: [
      { text: "Github", url: "https://github.com/MorariuT/2D-Raytracing-in-C" },
      { text: "Explanation", url: "./Projects/2D_SDL_raycaster.html" }
    ],
    media: {
      type: "image",
      src: "./media/ss1.png",
      alt: "2D Raycaster Screenshot"
    }
  },
  {
    title: "2D SDL Gravity Simulator",
    description: "A simple gravity simulator with planets, inspired by a youtube video.",
    size: "medium",
    links: [
      { text: "Github", url: "https://github.com/MorariuT/2D-C-Gravity-Simulator" },
      { text: "Explanation", url: "./Projects/2D_SDL_gravity.html" }
    ],
    media: {
      type: "video",
      src: "media/video.mp4"
    }
  },
  {
    title: "AI Work",
    description: "A big collection of jupyter notebooks prepared for different olympiads. It includes preparation for National/International AI Olympiads, from-scratch implementations, and theory. Updated every couple of months.",
    size: "medium",
    links: [
      { text: "Github", url: "https://github.com/MorariuT/ai_work" },
      { text: "Explore!", url: "./Projects/AI_Work.html" }
    ]
  },
  {
    title: "Evolution Simulator",
    description: "A simple genetic algorithm based on a Neural Network made for a presentation in my Biology class.",
    size: "medium",
    links: [
      { text: "Github", url: "https://github.com/MorariuT/Evolution-Simulator" },
      { text: "Explanation", url: "./Projects/Evolving_Simulator.html" }
    ],
    media: {
      type: "video",
      src: "media/video_evo.mp4"
    }
  }
];

// AI Work projects
const aiWorkProjects = {
  competitions: [
    {
      title: "IOAI 2025",
      description: "International Olympiad in Artificial Intelligence, 2025, Beijing China",
      url: "https://github.com/MorariuT/ai_work/tree/main/Competitions/IOAI%202025",
      size: "small"
    },
    {
      title: "IAIO 2024",
      description: "International Artificial Intelligence Olympiad, 2024, Riyadh, Saudi Arabia",
      url: "https://github.com/MorariuT/ai_work/tree/main/Competitions/IAIO%202024",
      size: "small"
    },
    {
      title: "NEOAI 2024",
      description: "AI Olympiad for Northern Europe Countries, 2024, Online",
      url: "https://github.com/MorariuT/ai_work/tree/main/Competitions/NEOAI%202024",
      size: "small"
    },
    {
      title: "APOAI 2025",
      description: "Asian Pacific AI Olympiad, 2025, Online",
      url: "https://github.com/MorariuT/ai_work/tree/main/Competitions/APOAI%202025",
      size: "small"
    }
  ],
  computerVision: [
    {
      title: "VAE trained on human faces",
      description: "A Variational Auto Encoder (+ residual paths) trained on human faces for one-to-one reproducibility.",
      url: "https://github.com/MorariuT/ai_work/blob/main/Computer%20VIsion/VAE_From_Scratch_Residual_Paths_1k.ipynb",
      size: "large"
    },
    {
      title: "Digit OCR",
      description: "OCR trained from scratch on MNIST Digits. Has CPU and GPU variants based on torch.",
      url: "https://github.com/MorariuT/ai_work/tree/main/Computer%20VIsion/Digit%20Recognizer%20MNIST",
      size: "small"
    },
    {
      title: "Water Meter Segmentation",
      description: "Detecting the area with digits on water meters using a UNet based on torch.",
      url: "https://github.com/MorariuT/ai_work/tree/main/Computer%20VIsion/Water%20Meter%20Segmentation",
      size: "small"
    },
    {
      title: "From Scratch MNIST Autoencoder",
      description: "An Autoencoder trained on the Fashion MNIST dataset that can reduce noise with remarkable accuracy.",
      url: "https://github.com/MorariuT/ai_work/blob/main/Computer%20VIsion/From_Scratch_Auto_Encoder_MNIST.ipynb",
      size: "medium"
    }
  ],
  nlp: [
    {
      title: "LSTM Text Generation",
      description: "A text-to-text model with a LSTM backbone trained on food reviews.",
      url: "https://github.com/MorariuT/ai_work/blob/main/Natural%20Language%20Processing/LSTM_Text_Generation.ipynb",
      size: "small"
    },
    {
      title: "Vaccination Tweet Classification",
      description: "Classifying tweets about vaccination topics.",
      url: "https://github.com/MorariuT/ai_work/blob/main/Natural%20Language%20Processing/Vaccination_Tweet_Classifier.ipynb",
      size: "small"
    }
  ],
  machineLearning: [
    {
      title: "Book Price Prediction",
      description: "Implementation of the classic book price prediction task.",
      url: "https://github.com/MorariuT/ai_work/blob/main/Machine%20Learning/Book_Price_Prediction.ipynb",
      size: "small"
    },
    {
      title: "Financial Inclusion In Africa",
      description: "A realistic problem taken from Zindi in preparation for IAIO 2024.",
      url: "https://github.com/MorariuT/ai_work/blob/main/Machine%20Learning/Financial_Inclusion_In_Affrica.ipynb",
      size: "small"
    },
    {
      title: "H1N1 Flu Vaccine Prediction",
      description: "The first AI problem ever solved!",
      url: "https://github.com/MorariuT/ai_work/blob/main/Machine%20Learning/Flu_Shot_Learnig_Predict_H1N1_and_Seasonal_Flu_Vaccines.ipynb",
      size: "small"
    },
    {
      title: "Smoker detection based on Bio signals",
      description: "Detecting if a person smokes based on their medical status.",
      url: "https://github.com/MorariuT/ai_work/blob/main/Machine%20Learning/Smoker_Detection_Based_on_Bio_Signals.ipynb",
      size: "small"
    }
  ],
  fromScratch: [
    {
      title: "Gradient Descent + All the flavors",
      description: "Implementations of various gradient descent methods in a simple class.",
      url: "https://github.com/MorariuT/ai_work/tree/main/Misc/Things%20From%20Scratch",
      size: "medium",
      subItems: [
        { text: "Simple Gradient Descent", url: "https://github.com/MorariuT/ai_work/blob/main/Misc/Things%20From%20Scratch/SimpleGradientDescent.ipynb" },
        { text: "Momentum Gradient Descent", url: "https://github.com/MorariuT/ai_work/blob/main/Misc/Things%20From%20Scratch/MomentumGradientDescent.ipynb" },
        { text: "Stochastic Gradient Descent", url: "https://github.com/MorariuT/ai_work/blob/main/Misc/Things%20From%20Scratch/StochasticGradientDescent.ipynb" },
        { text: "Weight Decay Gradient Descent", url: "https://github.com/MorariuT/ai_work/blob/main/Misc/Things%20From%20Scratch/WeightDecayGradientDescent.ipynb" }
      ]
    }
  ]
};
