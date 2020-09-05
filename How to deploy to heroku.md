# How to deploy a Python app, which uses git lfs, to Heroku

## Heroku account

In order to deploy this app or any other app to Heroku you will need an account.

You can register via the [Heroku website](https://www.heroku.com/).

After this you will need to download the [Heroku cli tool](https://devcenter.heroku.com/articles/heroku-cli#download-and-install):

```
curl https://cli-assets.heroku.com/install.sh | sh
```

Next, login to your heroku account from the cli:

```
heroku login
```

## Deployment

Finally it's time to deploy the project to heroku. Navigate to the project root. There you will need 2 files:
- Procfile, which specifies how to run the app
- run.py, a script which will normally start the app

Samples for such files can be found in this repo. [Procfile](https://github.com/luchev/disaster-response-pipeline/blob/master/Procfile) and [run.py](https://github.com/luchev/disaster-response-pipeline/blob/master/run.py). The Procfile uses `gunicorn` to start the Flask app. The startup script for the Flask app is in `run.py`. It's important that `run.py` doesn't call `app.run(host='0.0.0.0', port=3001, debug=True)` or any other `app.run()` directly - gunicorn will take care of that.

Next let's create the heroku app. Pick a name, for this project the name will be `disasterresponsepipeline` and run:

```
heroku create --buildpack https://github.com/raxod502/heroku-buildpack-git-lfs.git disasterresponsepipeline
```

The above script creates a heroku app with `disasterresponsepipeline` and a buildpack for `git lfs`. Read more what this buildpack does [here](https://elements.heroku.com/buildpacks/raxod502/heroku-buildpack-git-lfs).

Next, we will have to specify a git lfs repository from where Hiroku can download the large files. The repository is simply a link to Github and is specified in the following way (you can change the github link to your repo link):

```
heroku config:set HEROKU_BUILDPACK_GIT_LFS_REPO=https://github.com/luchev/disaster-response-pipeline
```

One last step is to make sure we have a way to tell Hiroku what python packages we will need installed on the host. To do so we can use a [virtual environment](https://docs.python.org/3/tutorial/venv.html) to isolate our project from our system and then install all the packages we need with pip. In the end to get a list of all packages we have we can use `pip freeze`. This step is not needed if you want to deploy a project which already uses virtual environment and has `requirements.txt`. Here's an example of what this would look like:

```
python3 -m venv my-virtual-environment-name
source my-virtual-environment-name/bin/activate
pip install pandas numpy ...
pip freeze > requirements.txt
```

We are finally ready to deploy our app. Make sure you have committed all changes you want and push to heroku. Use the flag --no-verify if you have problems with git lfs.

```
git push heroku master --no-verify
```

## Common problems

If your app fails to load you might need to add the `python` and `git-lfs` buildpacks. You have 2 options for this:
- Using the cli
    ```
    heroku buildpacks:add https://github.com/raxod502/heroku-buildpack-git-lfs -a <MY-PROJECT-NAME>
    ```
- Using the Heroku website. Simply navigate to your project settings and under Buildpacks click `Add buildpack`. There you can add `python` and `https://github.com/raxod502/heroku-buildpack-git-lfs`

If your git push is getting denied, make sure you have first pushed all your commits to your github repo and you're using the `--no-verify` flag when pushing to hiroku.
