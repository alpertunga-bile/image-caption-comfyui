# image-caption-comfyui

- [image-caption-comfyui](#image-caption-comfyui)
  - [Setup](#setup)
  - [Example Workflow](#example-workflow)
  - [Pretrained Image Caption Models](#pretrained-image-caption-models)
    - [Models](#models)
  - [Variables](#variables)
  - [Troubleshooting](#troubleshooting)
  - [Contributing](#contributing)
  - [Example Output](#example-output)

Image caption node for ComfyUI. You can load your image caption model and generate prompts with the given picture.

## Setup
- Clone the repository to the ```custom_nodes``` folder
- Run ComfyUI
- Place the folder which contains your model under the ```models/image_captioners``` folder
- Click ```Refresh``` button in ComfyUI, if it didn't work restart ComfyUI

- [x] Processor has to be in the folder

## Example Workflow

![basic_workflow](images/basic_workflow.png)

![basic_workflow_w_prompt_generator](images/basic_workflow_with_promp_generator.png)

![basic_workflow_w_prompt_generator_2](images/basic_workflow_with_prompt_generator_2.png)

## Pretrained Image Caption Models

- You can find the models in [this link](https://drive.google.com/drive/folders/1c21kMH6FTaia5C8239okL3Q0wJnnWc1N?usp=share_link)

- For to use the pretrained model follow these steps:
  - Download the model and unzip to ```models/image_captioners``` folder.
  - Click ```Refresh``` button in ComfyUI
  - Then select the image caption model with the node's ```model_name``` variable (If you can't see the generator, restart ComfyUI).

### Models

- female_image_caption_blip | **(Training In Process)**
    - Base model
    - using [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)

## Variables

|     Variable Names     | Definitions                                                                             |
| :--------------------: | :-------------------------------------------------------------------------------------- |
|     **model_name**     | Folder name that contains the model                                                     |
|   **min_new_tokens**   | The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt. |
|   **max_new_tokens**   | The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt. |
|     **num_beams**      | Number of steps for each search path                                                    |
| **repetition_penalty** | The parameter for repetition penalty. 1.0 means no penalty                              |

- For more information, follow [this link](https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/text_generation#transformers.GenerationConfig).
- Check [this link](https://huggingface.co/docs/transformers/v4.31.0/en/generation_strategies#text-generation-strategies) for text generation strategies.

## Troubleshooting

- If the below solutions are not fixed your issue please create an issue with ```bug``` label

## Contributing

- Contributions are welcome. If you have an idea and want to implement it by yourself please follow these steps:

  1. Create a fork
  2. Pull request the fork with the description that explaining the new feature

- If you have an idea but don't know how to implement it, please create an issue with ```enhancement``` label.

- [x] The contributing can be done in several ways. You can contribute to code or to README file.


## Example Output

![output_1](images/output_1.png)