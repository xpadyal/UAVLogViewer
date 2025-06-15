// Worker.js
// import MavlinkParser from 'mavlinkParser'
const mavparser = require('./mavlinkParser')
const DataflashParser = require('./JsDataflashParser/parser').default
const DjiParser = require('./djiParser').default

let parser
self.addEventListener('message', async function (event) {
    if (event.data === null) {
        console.log('got bad file message!')
    } else if (event.data.action === 'parse') {
        const data = event.data.file
        console.log('Data received in worker for parsing:', data);
        if (event.data.isTlog) {
            parser = new mavparser.MavlinkParser()
            parser.processData(data)
        } else if (event.data.isDji) {
            parser = new DjiParser()
            await parser.processData(data)
        } else {
            parser = new DataflashParser(true)
            parser.processData(data)
        }

    } else if (event.data.action === 'loadType') {
        if (!parser) {
            console.log('parser not ready')
        }
        parser.loadType(event.data.type.split('[')[0])
    } else if (event.data.action === 'trimFile') {
        parser.trimFile(event.data.time)
    }
})
