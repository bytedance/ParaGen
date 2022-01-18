namespace py Thrift

struct Request{
    // a json-dumped string for a batch of samples
    1: required string samples;
}

struct Response{
    // a json-dumped string for a batch of results
    1: string results;
    2: string debug_info;
    // message for this request, "Success" or others
    4: i32 code;
}

service Service{
    // infer the result score of title
    Response serve(1:Request req)
}
