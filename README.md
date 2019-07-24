# Coral Teachable Machine

Teachable machine allows you to quickly and interactively train a computer vision system to recognize objects simply by offering up examples to a camera and pressing one of 4 buttons to select the classification.  The project is a demonstration of the use of pre-trained embeddings for classification which obviates the need for retraining a whole network (potentially at the expense of accuracy).

As in previous teachable machine incarnations  ([Creative Lab](https://teachablemachine.withgoogle.com/) and [Tensorflow.js](https://beta.observablehq.com/@nsthorat/how-to-build-a-teachable-machine-with-tensorflow-js) notebook) this gadget immediately demonstrates the abstractive power deep neural networks can achieve. Even with a small number of examples the teachable machine can distinguish objects relatively independent of their exact orientation, etc. The UI is intuitive and immediately rewarding, while the embedded device responds quickly and with low latency. This sort of thing wasn’t possible even a few years ago on a powerful GPU but now we can have it in this tiny form factor.

### How it works:

Teachable machine relies on a pretrained image recognition network called MobileNet. This network has been trained on a vast number of images of 1000 different objects and has learned to recognize them (such as cats, dogs, cars, fruit, birds etc). In the process of doing so it has developed a semantic representation of the image that’s maximally useful to distinguish these classes. This internal representation can be used to quickly learn to distinguish new class the network has never seen, essentially a form of transfer learning.

Teachable machine uses a “headless” MobileNet, in which the last layer (which makes the final decision on the 1000 training classes) has been removed, exposing the output vector of the layer before. Teachable machine treats this output vector as a generic descriptor for a given camera image, called an embedding vector. The main idea is that semantically similar images will also give similar embedding vectors. Therefore, to make a classification, we can simply find the closest embedding vector of something we’ve previously seen and use it to determine what the image is showing now.
Each time the user presses a button the algorithm stores the embedding vector and its associated class (1-4) in a dictionary. Then, new objects’ embedding vectors can be compared to the library of saved embeddings and a simple classification can be achieved by finding the closest correspondence.


### More info

More info on the details of the algorithm are in [this detailed post](https://beta.observablehq.com/@nsthorat/how-to-build-a-teachable-machine-with-tensorflow-js)
. This project is a direct re-implementation of this earlier web version (See also [Creative Lab’s  original Teachable Machine](https://experiments.withgoogle.com/teachable-machine) for a web-hosted version to play with using your webcam)

## Installation

This project can be run on either a Raspberry Pi 3 with a Pi Camera or on the Coral DevBoard with
the Coral camera.

## On a Raspberry Pi 3:

### Get the Pi ready.

Boot your Pi and ensure that the camera is working correctly. A quick way to do
that is:
```shell
raspistill -v -o test.jpg
```
If it's not enabled you may need to enable it in ```raspi-config```.
Also update your system

```shell
sudo apt-get update
```


### Install the EdgeTPU

We need to install the Coral EdgeTPU libraries.
Follow the instructions at [https://coral.withgoogle.com/tutorials/accelerator/](https://coral.withgoogle.com/tutorials/accelerator/) to set up your accelerator.

After installation you can double check everything has worked:
(Re)Plug in your USB Accelerator using the supplied USB cable and run

```shell
python3 -c 'import edgetpu; print("OK")'
```

This command should print "OK" and not show any error messages if everything worked.


### Run the Teachable Machine

Install all the dependencies

```shell
sh install_requirements.sh
```

You can try out the teachable machine right away using your keyboard as input:

```shell
python3 teachable.py --keyboard
```

## On the Coral DevBoard:

On the Coral DevBoard the EdgeTPU libraries are already preinstalled. After attaching your
camera, all you have to do is:

```shell
sh install_requirements.sh
python3 teachable.py --keyboard
```

## Using the Teachable Machine

To operate our Teachable machine we need to teach it ! Let’s start adding class examples to the demo by pressing the buttons 1-4, which correspond to the 4 classes we can learn. 

First, I always like to reserve one of the classes for “background”. I usually just press '1' a few times before offering up any real examples to the camera.

Now hold an item over the camera and press one of the other buttons 2-4. This will save a snapshot of the image’s embedding linked to the button that was pressed. You can repeat this with another object and another colored button or add additional examples to the same class. It is recommended to add multiple examples of each class or object. For example shows it an apple in various orientations and distances each time pressing blue (do not hold down the button, instead do multiple individual presses while holding the object still during each press). Then hold a banana and give the tachable machine multiple examples of that in different orientations also. The example counter on the screen should go up accordingly every time you press a button.

While no button is pressed the demo will continuously interpret the current image and display the class it thinks the image belongs to. If you find an object misclassified, then add another example in the orientation that gets misclassified. It’s easy to get a sense if enough examples have been added or where more examples are needed.

To clear the memory press the 'c' key. Ctrl-C will quit the demo.

### Tips & Ideas

Lighting is important - if the overhead lighting is too bright the contrast the camera sees may be very poor. In that case provide some upwards lighting or set the demo on it’s side, or shield the glare from above.

Classification using embeddings has its limits but for simple classification tasks it works surprisingly well. Experiment with different things to recognize. Usually having a static background helps a lot. Try with:

  * Hands at different heights
  * Holding out different no of fingers (1,2,3,4)
  * Fruit
  * Chess pieces
  * Faces (it’s surprisingly good at this).
  * Lego blocks

### Extending the project

Teachable machine is merely a blank starting point which you can adapt to a variety of different uses. Comparing generic embeddings can be used in a variety of ways and are a very generic way to leverage the semantic recognition powers of a pre-trained network. The advantage is that you do not have to expensively retrain the network on thousands of images, but instead you directly teach the device as needed (especially when it gets it wrong, you add another training datapoint on the fly). The disadvantage is that it is somewhat less accurate, so for very high precision tasks you cannot go around true retraining, but in many cases working with embeddings can get you most of the way there.

## Imprinting method
Instead of the k-nearest neighbors algorithm we can also use an alternative 
algorithm to train the Teachable Machine on device, called Imprinting. 
To try it out call the alternate script below - the controls and UI are exactly the same as before.

```bash
sh run_imprinting.sh
```

Depending on the situation, one or the algorithm may work better, it’s worth 
experimenting. If you want to learn more about how imprinting works, take a 
look at https://coral.withgoogle.com/docs/edgetpu/retrain-classification-ondevice/


### Next steps

The project as it is just a shell, it lacks a true output. Think of ways the machine could indicate or communicate recognition of objects. In software you have a variety of options. It could play sounds ? It could send a tweet or an email or some other form of notification. For example you could try to teach it to recognize you and your housemates.

You could also simply log data and serve the log using a simple python webserver. This way you could get stats or graphs of object recognition over time. Who empties the dishwasher more often ?

Finally you can also trigger hardware events by using the same GPIO pins. For example a transistor could trigger a relay which could turn on another device. Perhaps the device could greet different people. Or the reading light could switch on when a book is placed in view. Or perhaps you want the sprinkler to chase away the cats and only the cats from your kid’s sandbox ?

