# Bonsai2 1000-character chat capability check

Date: 2026-09-19. Live endpoint: `http://127.0.0.1:9931/v1/chat/completions` on the server bound to `0.0.0.0:9931`.

The primary Korean prompt was 1,023 characters. It required unit conversion, stable-capacity reasoning, first-month cost arithmetic, four exact section titles, two uncertainty items, and a 900-1,100 character answer ending with a fixed sentence. Sampling used temperature 0.2, top-p 0.9, seed 42, and `reasoning_effort: none` after the low-reasoning trial exhausted its 1,400-token limit.

| Check | First answer | Correction turn | Result |
| --- | ---: | ---: | --- |
| Output characters | 1,312 | 1,316 | fail; required 900-1,100 |
| Required headings | 4/4 | 4/4 | pass |
| Fixed final sentence | pass | pass | pass |
| Finish reason | stop | stop | complete responses |
| Decode speed | 14.983 tok/s | 14.856 tok/s | measured |
| Arithmetic | C total mishandled | same error retained | fail |

Most unit conversions and capacities were correct: 120/min = 2/s, 300/min = 5/s, stable A/B/C capacities are 12.6/8.4/7.0 per second, thermally derated A is 10.71/s, and A+B first-month power is 375,840 won. The recommendation to buy cooling plus fast failover is plausible and totals 4,175,840 won for the first month.

The response listed C power as 116,640 won but failed to add it to the total. Correct all-option arithmetic is `8,600,000 + (3.8 kW * 720 h * 180 won) = 9,092,480 won`, above the 9,000,000 won budget. It also said demand exceeded every stable limit when the intended relation was that every stable capacity exceeded demand. The correction turn repeated the total and length failures after both were explicitly identified.

The GGUF Jinja template is loaded and `--jinja` is active. It supports `reasoning_effort` values `xhigh` (default), `medium`, `low`, and the server-level `none` value that disables thinking. A fresh `max` request returns HTTP 500 because the template rejects that value. The `low` trial generated all 1,400 allowed tokens without yielding a captured final response, so strict-length ordinary chat should use `none`; reasoning workloads need a supported value and a larger token budget. `/props` reports content-only output, reasoning format `none`, one slot, and context 65,536.

Raw evidence is under `C:/AI/bonsai2_27b/pp-fix/`: `bonsai-1000char-{request,response}-none.json`, `bonsai-1000char-correction-{request,response}.json`, `bonsai-1000char-props.json`, and `bonsai-template-max-{request,response}.json`. Preserve these in the final Alchemist measurement archive.
