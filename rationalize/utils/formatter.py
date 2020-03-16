# coding: utf-8


# Format class, ab_cd_1efg to AbCd1efg.
def format_class(name):
    return "".join(piece.capitalize() for piece in name.split("_"))
