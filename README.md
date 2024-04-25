# Image Super-Resolution with EDSR

This project implements a super-resolution model based on the Enhanced Deep Residual Networks (EDSR) architecture. It allows users to enhance the resolution of their images using a pre-trained EDSR model.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Make sure you have Python installed on your system. You can download it from [python.org](https://www.python.org/).

### Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   ```

2. Navigate to the project directory:
   ```bash
   cd your-repo
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Run the `app.py` script:
   ```bash
   python app.py
   ```

2. Once the application is running, use the "Choose file" button to select the image you want to enhance.

3. After selecting the file, a prompt will appear asking you to select the location where you want the processed file to be stored. Choose your desired location and confirm.

4. The application will process the image using the EDSR model and save the enhanced image to the specified location.

## Acknowledgments

- This project is based on the Enhanced Deep Residual Networks (EDSR) architecture, developed by Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee. More information about EDSR can be found in their [paper](https://arxiv.org/abs/1707.02921).
- Special thanks to the authors of the pre-trained EDSR model used in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.