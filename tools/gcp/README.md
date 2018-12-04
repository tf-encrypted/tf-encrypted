# Experimenting on Google Cloud Platform

A few tools for making it slightly easier to run experiments on the Google Cloud Platform.

All steps here assume that the [Cloud SDK](https://cloud.google.com/sdk/) has already been installed (for macOS this may be done via e.g. Homebrew: `brew cask install google-cloud-sdk`) and a project set up.

It also assumes that you are running the commands from this directory (`cd tools/gcp/` from the project root.)

## Base image

We first create a base image with all needed software for future GCP instances running tf-encrypted. To that end we start with a template instance whos disk turns into the image.

Run the following to create the template instance
```shell
gcloud compute instances create tfe-template \
    --machine-type=n1-standard-4 \
    --image=ubuntu-minimal-1804-bionic-v20181203 \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=20GB
```
and this to afterwards install the needed software
```shell
gcloud compute ssh tfe-template < debian-install.sh
```

Then, to create the image, we run
```shell
gcloud compute instances stop tfe-template
gcloud compute images create tfe-image --source-disk tfe-template
```
and can optionally finally [delete the template instance](#cleaning-up) to save on cost.

With this image we can then easily create more instances with
```shell
gcloud compute instances create \
    INSTANCE_NAMES \
    --image tfe-image \
    --machine-type=n1-standard-4
```
or simply use the [`create`](./create) script.

### Cleaning up

To clean up the template instance and its disk we run
```shell
gcloud compute instances delete tfe-template --quiet
gcloud compute disks delete tfe-template --quiet
```
which can be done as soon as the image was been created without breaking anything.

If we wish to clean up fully the base image can also be deleted with
```shell
gcloud compute images delete tfe-image --quiet
```
If you encountered an error when creating the image tfe-image from the tfe-template, it's probably because you already had the image tfe-image created. For this reason you might have to run the command above as well.
