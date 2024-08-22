# Server

This directory host an updated version of `server-service` abstraction from the one used by HAKES-search, which exposes index serving logic at the service level. The new abstraction should follow:

* Server: libuv and http handling
* Service: bridges the worker logic exposed by Handle to the `OnWork` callback in server
* Derived class of Worker: provide the implementation of handling a request.
