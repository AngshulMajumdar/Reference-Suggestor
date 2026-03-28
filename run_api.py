from reference_suggester_api.app import create_app

app = create_app()

if __name__ == '__main__':
    import os
    import uvicorn

    host = os.environ.get('REFSUGGEST_HOST', '0.0.0.0')
    port = int(os.environ.get('REFSUGGEST_PORT', '8010'))
    uvicorn.run('run_api:app', host=host, port=port, reload=False)
