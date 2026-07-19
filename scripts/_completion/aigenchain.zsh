#compdef aigenchain aigenchain-backup aigenchain-calendar aigenchain-contacts aigenchain-cookbook aigenchain-docs aigenchain-gallery aigenchain-mail aigenchain-mcp aigenchain-memory aigenchain-notes aigenchain-personal aigenchain-preset aigenchain-research aigenchain-sessions aigenchain-signature aigenchain-skills aigenchain-tasks aigenchain-theme aigenchain-webhook
# Zsh tab-completion for the aigenchain umbrella + sub-CLIs.
#
# Drop in any directory on $fpath, e.g.:
#     fpath=(/path/to/aigenchain-ui/scripts/_completion $fpath)
#     autoload -U compinit; compinit
#
# Then `aigenchain <tab>` completes subcommands; `aigenchain mail <tab>`
# completes mail subcommands; `aigenchain-mail <tab>` works the same.

_aigenchain_scripts_dir() {
    local self="${(%):-%x}"
    while [[ -L "$self" ]]; do self="$(readlink "$self")"; done
    cd "${self:h}/.." && pwd
}

typeset -gA _aigenchain_subs

_aigenchain_refresh() {
    _aigenchain_subs=()
    local dir="$(_aigenchain_scripts_dir)"
    local py="$dir/../venv/bin/python"
    [[ -x "$py" ]] || py="$(command -v python3)"
    local f sub help_out commands
    for f in "$dir"/aigenchain-*; do
        [[ -x "$f" ]] || continue
        case "$f" in
            *.bak|*.pyc|*.pre-*) continue ;;
        esac
        sub="${${f:t}#aigenchain-}"
        help_out=$("$py" "$f" --help 2>/dev/null) || continue
        commands=$(echo "$help_out" | grep -oE '\{[a-z0-9_,-]+\}' | head -1 \
            | tr -d '{}' | tr ',' ' ')
        _aigenchain_subs[$sub]="$commands"
    done
}

_aigenchain() {
    [[ ${#_aigenchain_subs} -eq 0 ]] && _aigenchain_refresh

    local cmd="${words[1]}"

    if [[ "$cmd" == "aigenchain" ]]; then
        if (( CURRENT == 2 )); then
            local -a subs=(${(k)_aigenchain_subs} help)
            _describe 'subcommand' subs
            return
        fi
        local sub="${words[2]}"
        if [[ "$sub" == "help" ]] && (( CURRENT == 3 )); then
            local -a subs=(${(k)_aigenchain_subs})
            _describe 'subcommand' subs
            return
        fi
        if (( CURRENT == 3 )); then
            local -a sc=(${(s/ /)_aigenchain_subs[$sub]})
            _describe 'command' sc
            return
        fi
        return
    fi

    # aigenchain-foo <tab>
    local sub="${cmd#aigenchain-}"
    if (( CURRENT == 2 )); then
        local -a sc=(${(s/ /)_aigenchain_subs[$sub]})
        _describe 'command' sc
        return
    fi
}

_aigenchain "$@"
