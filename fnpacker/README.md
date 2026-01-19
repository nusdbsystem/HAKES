# FnPacker

FnPacker is a middleware to enable packing of embedding models with different model parameters served with a unified set of endpoint to improve resource utilization. It has the accept a same endpoint creation, invoke and deletion API but manages a pool of endpoint behind the scene and for the client side, the API is the same as without it.

## Organization

* `fnpack`: internal of FnPacker
* `fpcli`: sample go client for FnPacker
* `req`: defines the messaging protocol for clients calling FnPacker
* `scli`: interface FnPacker and serverless platform (implemented for OpenWhisk)
* `main.go`: the FnPacker server
