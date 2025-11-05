# blabin
Adaptive agent for helping me learn french faster


## Auth to google cloud

The application is set up to use application-default credentials for authenticating to google cloud resources. In order to use gcloud tools, you must auth with gcloud cli (installed in dev container).

```sh
gcloud init # will open up oauth
gcloud auth application-default login
```
