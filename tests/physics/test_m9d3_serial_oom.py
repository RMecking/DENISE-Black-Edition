"""Serial ownership sweep, including the new surface reverse scratch site."""
import os,shutil,subprocess

def test_serial_surface_all_reachable_failures(tmp_path,repository_root):
    flags=['-std=c99','-g','-O2','-Wall','-Wextra','-Werror','-pedantic']
    env=os.environ.copy()
    if env.get('DENISE_M9D3_SERIAL_SANITIZE')=='1':
        flags+=['-O1','-fsanitize=address,undefined','-fno-omit-frame-pointer','-fno-pie','-no-pie']
        env.update(ASAN_OPTIONS='detect_leaks=1:abort_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
    executable=tmp_path/'serial-ledger'
    subprocess.run([shutil.which('cc'),*flags,'-I',str(repository_root/'include'),
        str(repository_root/'tests/utilities/m9d3_serial_oom.c'),
        str(repository_root/'src/PSV/elastic_psv_born.c'),'-Wl,--wrap=calloc','-Wl,--wrap=free','-lm','-o',str(executable)],check=True)
    result=subprocess.run([str(executable)],env=env,capture_output=True,text=True,timeout=120)
    (tmp_path/'serial-ledger.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'zero_outstanding=YES' in result.stdout and 'PASS' in result.stdout
    print(result.stdout)
