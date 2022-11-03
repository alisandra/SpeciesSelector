### Notes to self on how to handle changing code / db during active projects
Not pretty, but works for now

```
nnictl view <split0ID>
nnictl view <split1ID> -p 8081
```
`rsync -rv <ori_wd> <tester_wd>`

`spselec reset-wd --working-dir <tester_wd>`

`python scripts/one_off_migrate.py --old <tester_wd>/spselec.sqlite3 --new <tester_wd>/test.sqlite3`

`mv <tester_wd>/test.sqlite3 <tester_wd>/spselec.sqlite3`
