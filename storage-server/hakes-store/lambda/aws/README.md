# Create Lambda Function on AWS

## Build docker image

```json
cd hakes-store/lambda/aws/
go mod tidy
go get github.com/aws/aws-lambda-go/lambda
GOOS=linux GOARCH=amd64 go build -o main lambda.go

docker build -t hakeskv:test .
```

## Create ECR and push docker image

```json
# create an ecr repo on aws
aws ecr create-repository --repository-name $image_name --image-scanning-configuration scanOnPush=true --region us-east-1

# tag the docker image with your own aws account
docker tag $image_name:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/$image_name:latest

# login aws ecr
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# push docker image
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/$image_name:latest
```

## Create function on AWS Lambda using the image

* Go to AWS Lambda page, create a function and select the "Container image" option
* Select the image just pushed to AWS ECR
* Select an execution role that has AWS S3 permissions
* Create function

## Test the function

* On AWS Lambda, go to "Test" page, and input the following JSON, make sure
that "TablePath" is the S3 Bucket name, and there are `0.sst` and `1.sst` files in this path

```json
    {
      "OutTablePrefix": "",
      "TablePath": <your bucket>,
      "TopTables": [
        {
          "Tid": "0.sst",
          "Size": 96577,
          "InSstCache": false
        }
      ],
      "BotTables": [
        {
          "Tid": "1.sst",
          "Size": 96583,
          "InSstCache": false
        }
      ],
      "ThisLevel": 0,
      "NextLevel": 1,
      "Opts": {
        "TableSize": 2097152,
        "BlockSize": 4096,
        "BloomFalsePositive": 0.01,
        "ChkMode": 0,
        "Compression": 1,
        "ZstdCompressionLevel": 1
      },
      "NumVersionsToKeep": 1,
      "HasOverlap": true,
      "DiscardTs": 1,
      "DropPrefixes": null,
      "UseSstCache": false
    }
```

* Click Test and obtain the following result, then the test is correct.

```json
    "{\"Success\":true,\"NewTables\":[{\"Tid\":\"000001.sst\",\"Size\":0,\"InSstCache\":false}]}"
```
