from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc)



def stamp(value):
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds")



def parsed(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))

