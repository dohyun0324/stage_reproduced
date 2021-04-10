#!/usr/bin/env bash
fileId=1GnknXfs9qKE-WVaUgUeKfCTLHjyzqCHG
fileName=tvqa_plus_stage_features_new.tar.gz
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName}
