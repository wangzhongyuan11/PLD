import hashlib,json,pathlib,stat,zipfile
archive=pathlib.Path("/home/xilinx/PLD_Perf10_Deploy_20261020.zip")
root=pathlib.Path("/home/xilinx/jupyter_notebooks/pld/releases/pld_perf10_20261020")
assert hashlib.sha256(archive.read_bytes()).hexdigest()=="1127b5e3d208ce3fa61fbd989228858d94cdfadfe734bbcafc38dc04138b1c7b"
assert not root.exists(),"New independent release directory required"
with zipfile.ZipFile(archive) as z:
    names=set(); folded=set()
    for i in z.infolist():
        p=pathlib.PurePosixPath(i.filename)
        assert not p.is_absolute() and ".." not in p.parts and "\\" not in i.filename and ":" not in i.filename
        assert not stat.S_ISLNK(i.external_attr>>16) and i.filename.casefold() not in folded
        folded.add(i.filename.casefold());names.add(i.filename)
    manifest=json.loads(z.read("payload_sha256.json"))
    assert set(manifest)|{"payload_sha256.json"}==names
    for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    root.mkdir(mode=0o700)
    z.extractall(root)
print("INSTALLED_VERIFIED",len(manifest),root)
