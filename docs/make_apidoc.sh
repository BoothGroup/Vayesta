sphinx-apidoc -o source/apidoc -t templates ../vayesta/ \
`# Exclude:` \
../vayesta/core/qemb/rdm.py \
../vayesta/core/qemb/expval.py \
../vayesta/core/vpyscf/** \
../vayesta/ewf/amplitudes.py \
../vayesta/ewf/icmp2.py \
../vayesta/ewf/rdm.py \
../vayesta/ewf/urdm.py \
../vayesta/misc/gmtkn55** \
../vayesta/tests/**
