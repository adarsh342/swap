#!/usr/bin/env python3

from adarsh import core

if __name__ == '__main__':
    if not core.pre_check():
        core.destroy()
    core.parse_args()
    core.limit_resources()
    core.start()
