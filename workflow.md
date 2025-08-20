# Workflow git cheatsheet

- main - production version
- dev    - development version

Workflow:
1. develop on `dev` branch
2. after testing, sync `dev` with `main`
3. use `main` for production deployments

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

# 2. Switch to main branch
git checkout main

# 3. Merge dev branch into main
git merge dev

# 4. Push updated main to remote
git push origin main

# 5. Now can deploy main branch to production environment
```

