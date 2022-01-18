echo 'Running autoflake ...'
find tools -type f -name "*.py" | xargs autoflake --in-place --remove-all-unused-imports --ignore-init-module-imports 
# autoflake --in-place --remove-all-unused-imports --ignore-init-module-imports --recursive ./

echo 'Running isort ...'
find tools -type f -name "*.py" | xargs isort

echo 'Running autopep8 ...'
find tools -type f -name "*.py" | xargs autopep8 -i