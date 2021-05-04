# Model definition of the kinematic prediction transformer.

All components are in the code model.py, the model is the class EncoderDecoder, which can be instantiated by instantiate an object of this class.

This model is made up of a encoder, a decoder and a generator which are defined by the class Encoder, Decoder, and generator respectively

- Encoder consists of N encoder layer which is defined by the class EncoderLayer. 
The input of the encoder is the embedding of the input info (kinematics, task, state, video frames) plus a positional encoding.
The output of the decoder is a feature representation of the input info.

- Decoder consists of N decoder layer which is defined by the class decoder layer.
The input of the encoder are the feature representation given by the encoder and the embedding of the target info ( kinematics) plus a positional encoding.
the output of the decoder is a feature representation.

- The generator is a MLP for the kinematics prediction.
The input of the generator is the eature representation given by the decoder.
the output of the generator is the desired kinematics.

The detail of the EncoderLayer, DecoderLayers and Embeddings are in the comments of the code.
