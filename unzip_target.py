from zipfile import ZipFile

archive = ZipFile('./selected.zip')

ids = ['1000', '1095']
allowed_prefixes = tuple([f'{id}/' for id in ids])
exclusions = ['.html', 'parts_render']

for name in archive.namelist():
    if name.startswith(allowed_prefixes) and not any(exclusion in name for exclusion in exclusions):
        archive.extract(name, 'selected')
