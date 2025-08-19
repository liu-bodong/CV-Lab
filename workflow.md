# Workflow git cheatsheet

- master - production version
- dev    - development version

Workflow:
1. develop on `dev` branch
2. after testing, sync `dev` with `master`
3. use `master` for production deployments

Everyday development flow:

```
# 1. Start from clean dev branch
git checkout dev
git pull origin dev

# 2. Make code changes...

# 3. Commit changes to dev branch
git add .
git commit
git push origin dev

# 4. Test on dev branch...
```

Ready to publish to production env:
```
# 1. Make sure dev branch is up to date
git checkout dev
git pull origin dev

# 2. Switch to master branch
git checkout master

# 3. Merge dev branch into master
git merge dev

# 4. Push updated master to remote
git push origin master

# 5. Now can deploy master branch to production environment
```

