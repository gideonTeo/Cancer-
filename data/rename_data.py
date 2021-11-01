import os

# takes in a dirname and renames & indexes them to the acronym (acro), takes about 87s
def image_rename(dirname,acro):
    assert isinstance(dirname, str)
    assert isinstance(acro, str)
    for i, filename in enumerate(os.listdir(dirname)):
        os.rename(dirname + "/" + filename, dirname + "/" + str(acro)+str(i) + ".jpg")
root= os.getcwd()
image_rename(root + "/data/archive/MSS_JPEG", "MSS_")
image_rename(root + "/data/archive/MSIMUT_JPEG", "MSI_")
