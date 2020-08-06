# FLIM - Feature Learning from Image Markers

Feature Learning from Image Markers is a technique to learn the filter of convolutional neural networks' feature extractors from user-drawn image markers.

This package provides an implementation of this technique powered by Pytorch, Scikit-learn, and Scikit-image.

To install it, go to the folder where the package it and run the command

 ```
 pip install . 
 ```

 To install dependencies run the command

 ```
 pip install -r requirements.txt
 ```

To build the package API reference documentation, go to the folder `docs` and run the command

```
make html
```

You can run a simple HTTP server to serve the documetation page.

```
python -m http.server
```

Go to [localhost:8000](localhost:8000) and you navigate through the documentation.