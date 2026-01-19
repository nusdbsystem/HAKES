# TFLM Runtime

Currently we use a release build of tflm static library and the a example that uses the person detection model in the tflite-micro example directory. (We also refer the codes in tensorflow-lite-sgx repository when creating this example).

It only demonstrate the use of the Int8 quatized person detection CNN model. We shall consider generalize it to be a runtime that accept some user input to pick the proper interpreter and load the model and data properly.
