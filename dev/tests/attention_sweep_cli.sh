#!/bin/sh
# Invalid benchmark arguments must fail before a device or metallib is opened.
set -eu
binary=$1
work=$(mktemp -d "${TMPDIR:-/tmp}/splash-sweep-cli.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

reject() {
    expected=$1
    shift
    status=0
    "$binary" "$work/missing.metallib" "$@" >"$work/out" 2>"$work/err" || status=$?
    if [ "$status" -ne 70 ] || ! grep -Fq -- "$expected" "$work/err"; then
        echo "attention-sweep accepted or misclassified arguments: $*" >&2
        cat "$work/err" >&2
        exit 1
    fi
    test ! -s "$work/out"
}

for value in 0 5 100 -1 4294967296 4294967297 1x '' '1,' ',1' '1,,4'; do
    reject '--lanes requires integers' --lanes "$value"
done
for value in 0 -1 4294967296 1x ''; do
    reject '--repeat requires integers' --repeat "$value"
done
for value in -1 262144 4294967296 10x '' '0,'; do
    reject '--histories requires integers' --histories "$value"
done
for value in typo '' '27b,' '27b,typo' ',35b'; do
    reject '--shapes takes 27b or 35b' --shapes "$value"
done
for value in 0 129 -1 4294967296 1x ''; do
    reject '--verify-split-floor requires integers from 1 to 128' \
        --verify-split-floor "$value" --verify-min-pages 4
done
for value in 0 8193 -1 4294967296 1x ''; do
    reject '--verify-min-pages requires integers from 1 to 8192' \
        --verify-split-floor 8 --verify-min-pages "$value"
done
reject '--verify-split-floor and --verify-min-pages go together' --verify-split-floor 8
reject '--verify-split-floor and --verify-min-pages go together' --verify-min-pages 4
for option in --histories --lanes --repeat --shapes --phases --compare-metallib \
        --tile --verify-split-floor --verify-min-pages; do
    reject "$option requires a value" "$option"
done
reject '--phases takes all, both, verify, prefill or draft' --phases typo
reject '--tile takes mpp or register' --tile typo
reject 'unknown option --unknown' --unknown 1
# Valid values must reach the final sentinel, still without opening Metal.
reject 'unknown option --sentinel' \
    --lanes 1,2,3,4 --histories 0,2048,131072 --repeat 1 \
    --shapes 27b,35b --tile register --verify-split-floor 128 \
    --verify-min-pages 1 --sentinel 1
reject 'unknown option --sentinel' \
    --verify-split-floor 1 --verify-min-pages 8192 --sentinel 1
echo 'attention-sweep CLI validation: PASS'
