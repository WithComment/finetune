## Finetune
### Finetuning dataset requirements


## Evaluation
### Evaluation dataset requirements
Each entry should have
1. `id`
2. `question`
3. `answer` for multiple choice questions, the label, e.g., `A`, `B`, etc. For yes/no questions, `Y` and `N`.
4. `media` list of images and/or videos used in the question. Videos must be relative path from `args.data_dir.`
Image options
    - A relative path from `args.data_dir;`
    - URL;
    - `PIL.Image.Image;`
    - Base64 encoded image.
5. `options` required for multiple choice. a list containing the options.

#### Example
```json
[
  {
    "question": "Some yes/no question",
    "answer": "Y",
    "media": [
      "path/to/image.jpg",
      "http://path/to/image.jpg",
      "data:image;base64...",
      <PIL.Image.Image>
    ]
  },
  {
    "question": "What is the answer to the ultimate question",
    "answer": "A",
    "media": ["path/to/video.mp4"],
    "options": [
      "A: 42",
      "B: -1/12",
      "C: pi^2/6"
    ]
  }
]
```